import torch
import torch.nn as nn
from transformers import BertModel
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from Prompting.PromptedBert import PromptedBertModel


class MultitaskBERTModel(nn.Module):
    def __init__(self, config, pretrained_weights=None):
        super(MultitaskBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)

        # Freeze the first half of BERT layers to stabilize training
        for name, param in self.bert.named_parameters():
            if "encoder.layer." in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < 6:
                    param.requires_grad = False

        # Improved time embedding layer
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Initialize time embedding weights properly
        with torch.no_grad():
            self.time_embedding[0].weight.data.uniform_(-0.1, 0.1)
            self.time_embedding[0].bias.data.zero_()

        self.dropout = nn.Dropout(p=0.1)
        self.shared_hidden = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.shared_activation = nn.ReLU()

        # Next-Activity head (maps [batch, seq_len, hidden_dim] -> [batch, seq_len, num_activities])
        self.next_activity_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_activities)
        )

        # Outcome head (maps [batch, hidden_dim] -> [batch, num_outcomes])
        self.outcome_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(config.embedding_dim + config.loop_feat_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_outcomes)
        )

        # Build our custom BERT with MHSA injection (with modular prompts)
        self.prompted_bert = PromptedBertModel(
            config,
            pretrained_weights=pretrained_weights,
            enable_g_prompt=True,  # Enable/disable G-Prompts here for ablation studies
            enable_e_prompt=True  # Enable/disable E-Prompts here for ablation studies
        )

        if torch.cuda.is_available():
            self.prompted_bert = self.prompted_bert.to(torch.cuda.current_device())

    def forward(self, input_ids, attention_mask, times, loop_features=None, customer_type=None):
        """
        Args:
            input_ids:       [batch_size, seq_len]        Token/Activity IDs
            attention_mask:  [batch_size, seq_len]        1=valid token, 0=padding
            customer_type:   [batch_size] or None         For concept drift adaptation
            times:           [batch_size, seq_len]        Normalized time values

        Returns:
            next_activity_logits: [batch_size, seq_len, config.num_activities]
            outcome_logits:       [batch_size, config.num_outcomes]
        """

        # 1) Use PromptedBertModel to inject G/E-Prompts inside the self-attention layers
        #    This returns last_hidden_state of shape [batch_size, seq_len, hidden_dim]

        # Validate time inputs
        if torch.isnan(times).any():
            raise ValueError(f"Found {torch.isnan(times).sum().item()} NaN values in input times")
        if torch.isinf(times).any():
            raise ValueError(f"Found {torch.isinf(times).sum().item()} Inf values in input times")
        # Ensure times have the correct shape
        if times.dim() != 2:
            raise ValueError(f"Expected times tensor to have 2 dimensions, got {times.dim()}")

        # Memory optimization
        torch.cuda.empty_cache()

        # Create time mask that considers both padding (attention_mask) and valid times
        time_mask = attention_mask.unsqueeze(-1).float()
        time_mask = time_mask * (~torch.isnan(times)).float().unsqueeze(-1)

        # Keep original times but mask NaN positions
        times_clean = times.clone().unsqueeze(-1)

        # Get time embeddings with proper masking
        time_embeddings = self.time_embedding(times_clean)

        # Apply mask to zero out invalid positions while preserving gradient paths
        time_embeddings = time_embeddings * time_mask.expand_as(time_embeddings)

        # Forward through prompted BERT
        last_hidden_state = self.prompted_bert(
            input_ids,
            attention_mask=attention_mask,
            times=times,
            customer_type=customer_type
        )

        # 2) Next-Activity Prediction (per-step)
        seq_output = self.dropout(last_hidden_state)
        next_activity_logits = self.next_activity_head(seq_output)
        # Shape => [batch_size, seq_len, num_activities]

        # Modify next activity logits to force padding predictions to be padding
        padding_mask = (input_ids == 0).unsqueeze(-1).expand_as(next_activity_logits)
        next_activity_logits = next_activity_logits.masked_fill(padding_mask, float('-inf'))
        # Force the padding token (0) logits to be high where padding exists
        next_activity_logits[:, :, 0] = next_activity_logits[:, :, 0].masked_fill(padding_mask[:, :, 0], float('inf'))

        # 3) Outcome Prediction (single label per sequence)
        #    Average pooling across valid tokens, determined by attention_mask
        mask_expanded = attention_mask.unsqueeze(-1).float().expand_as(last_hidden_state)
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)  # [batch_size, hidden_dim]
        sum_mask = mask_expanded.sum(dim=1).clamp_min(1e-9)  # [batch_size, hidden_dim]
        pooled_output = sum_embeddings / sum_mask  # [batch_size, hidden_dim]

        # Convert to hidden dimension
        pooled_output = self.shared_hidden(pooled_output)
        pooled_output = self.shared_activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # Concatenate loop features if provided
        # Before concatenation, ensure both tensors have same dimensions
        if loop_features is not None:
            # Ensure loop_features has shape [batch_size, feature_dim]
            if loop_features.dim() == 1:
                loop_features = loop_features.unsqueeze(0)
            pooled_output = torch.cat([pooled_output, loop_features], dim=1)

        outcome_logits = self.outcome_head(pooled_output)
        # Shape => [batch_size, config.num_outcomes]

        return next_activity_logits, outcome_logits


def compute_class_weights(dataloader, device, num_activities, num_outcomes):
    # Compute frequencies of each class for both tasks
    activity_counts = torch.zeros(num_activities).long()
    outcome_counts = torch.zeros(num_outcomes).long()

    # Switch to CPU for counting frequencies to avoid overhead
    for batch in dataloader:
        activity_labels = batch['next_activity'].view(-1)  # Flatten next_activity labels
        outcome_labels = batch['outcome']  # Outcome labels

        for a in activity_labels:
            if a != -1:  # Ignore padding tokens
                activity_counts[a.item()] += 1
        for o in outcome_labels:
            outcome_counts[o.item()] += 1

    # Compute weights = 1 / frequency (normalized)
    activity_weights = 1.0 / (activity_counts.float() + 1e-9)
    activity_weights = activity_weights / activity_weights.sum() * len(activity_counts)

    outcome_weights = 1.0 / (outcome_counts.float() + 1e-9)
    outcome_weights = outcome_weights / outcome_weights.sum() * len(outcome_counts)

    return activity_weights.to(device), outcome_weights.to(device)


class DynamicWeightedLoss:
    def __init__(self, num_tasks=2):
        self.prev_losses = [1.0] * num_tasks
        self.weights = [0.5] * num_tasks

    def update(self, current_losses):
        # Compute relative improvement
        improvements = [prev / curr if curr > 0 else 1.0
                        for prev, curr in zip(self.prev_losses, current_losses)]
        total_imp = sum(improvements)

        # Update weights - give more weight to tasks that improve less
        self.weights = [imp / total_imp for imp in improvements]
        self.prev_losses = current_losses
        return self.weights


# In train_model():
loss_weighter = DynamicWeightedLoss()


def train_model(model, dataloader, optimizer, device, config, num_epochs=5, print_batch_data=True,
                accumulation_steps=4):
    """
    Fine-tune the multitask model with class weights for outcome prediction.
    Args:
        model: MultitaskBERTModel instance.
        dataloader: PyTorch DataLoader containing the fine-tuning dataset.
        optimizer: Optimizer for training.
        device: Training device (CPU or GPU).
        config: Config object with num_activities & num_outcomes, etc.
        num_epochs: Number of training epochs.
        accumulation_steps: Number of batches to accumulate gradients before an optimization step.
    """

    # Compute class weights from dataset
    # This may require one pass over the dataloader
    # We'll create a temporary dataloader iteration to gather frequencies.
    # IMPORTANT: Ensure that this does not disrupt your main training. If needed, recreate dataloader or cache data.
    model = model.to(device)
    activity_weights, outcome_weights = compute_class_weights(dataloader, device, config.num_activities,
                                                              config.num_outcomes)

    # Separate loss functions for each task with computed weights
    next_activity_loss_fn = nn.CrossEntropyLoss(weight=activity_weights)
    outcome_loss_fn = nn.CrossEntropyLoss(weight=outcome_weights)

    model.train()

    # Compute class weights only for outcome prediction task
    outcome_labels = []
    for batch in dataloader:
        outcome_labels.extend(batch['outcome'].numpy())

    # Calculate class weights for outcome prediction
    outcome_classes = np.unique(outcome_labels)
    outcome_weights = compute_class_weight('balanced',
                                           classes=outcome_classes,
                                           y=outcome_labels)

    # Convert weights to PyTorch tensor
    outcome_weights = torch.tensor(outcome_weights, dtype=torch.float).to(device)

    # Create loss functions:
    # Standard CE for next activity prediction
    next_activity_loss_fn = nn.CrossEntropyLoss().to(device)
    # Weighted CE for outcome prediction
    outcome_loss_fn = nn.CrossEntropyLoss(weight=outcome_weights).to(device)

    print(f"Outcome class weights: {outcome_weights.cpu().numpy()}")

    for epoch in range(num_epochs):
        total_loss = 0
        next_activity_total_loss = 0
        outcome_total_loss = 0

        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                            desc=f"Epoch {epoch + 1}/{num_epochs}")

        optimizer.zero_grad()
        for step, batch in progress_bar:

            # Validate batch data
            if torch.isnan(batch['times']).any():
                raise ValueError(f"NaN values found in batch times: {torch.isnan(batch['times']).sum().item()}")

            input_ids = batch['trace'].to(device)
            attention_mask = batch['mask'].to(device)
            customer_types = batch['customer_type'].to(device)
            loop_feats = batch['loop_features'].to(device)  # ← Get loop features
            times = batch['times'].to(device)
            next_activity_labels = batch['next_activity'].to(device)
            outcome_labels = batch['outcome'].to(device)

            if print_batch_data and epoch == 0 and step == 0:
                print("=== First Batch Data ===")
                print("Input IDs (trace):", input_ids)
                print("Relative Time:", times)
                print("Attention Mask:", attention_mask)
                print("Customer Types:", customer_types)
                print("Outcome Labels:", outcome_labels)
                print("========================")

            # Forward pass
            next_activity_logits, outcome_logits = model(input_ids, attention_mask, times, loop_feats, customer_types)

            # --- Next Activity Loss ---
            # Flatten (batch_size * seq_len) to align with cross-entropy:
            next_activity_logits_2d = next_activity_logits.view(-1, config.num_activities)
            next_activity_labels_1d = next_activity_labels.view(-1)

            # We only want valid positions (not padded). Flatten the attention_mask:
            valid_mask = attention_mask.view(-1).bool()
            # Add padding mask
            padding_mask = (input_ids != 0).view(-1).bool()
            # Combine masks to only consider non-padding valid positions
            final_mask = valid_mask & padding_mask

            valid_activity_logits = next_activity_logits_2d[final_mask]
            valid_activity_labels = next_activity_labels_1d[final_mask]

            next_activity_loss = next_activity_loss_fn(valid_activity_logits, valid_activity_labels)

            # --- Outcome Loss ---
            # One outcome per sequence, so no flattening.
            # outcome_logits: [batch_size, num_outcomes], outcome_labels: [batch_size]
            outcome_loss = outcome_loss_fn(outcome_logits, outcome_labels)

            # Get dynamic weights based on current losses
            task_weights = loss_weighter.update([next_activity_loss.item(),
                                                 outcome_loss.item()])

            # Combined loss with dynamic weights
            loss = (task_weights[0] * next_activity_loss +
                    task_weights[1] * outcome_loss) / accumulation_steps

            # Backward pass
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            # Track losses
            total_loss += loss.item() * accumulation_steps
            next_activity_total_loss += next_activity_loss.item()
            outcome_total_loss += outcome_loss.item()

            # Update progress bar with both losses
            progress_bar.set_postfix({
                'total_loss': f"{loss.item():.4f}",
                'next_act_loss': f"{next_activity_loss.item():.4f}",
                'outcome_loss': f"{outcome_loss.item():.4f}"
            })

        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        avg_next_activity_loss = next_activity_total_loss / len(dataloader)
        avg_outcome_loss = outcome_total_loss / len(dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs} completed:")
        print(f"Average Total Loss: {avg_loss:.4f}")
        print(f"Average Next Activity Loss: {avg_next_activity_loss:.4f}")
        print(f"Average Outcome Loss: {avg_outcome_loss:.4f}")