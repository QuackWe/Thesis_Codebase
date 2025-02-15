import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from Prompting.PromptedBert import PromptedBertModel
from FeatureEngineering import analyze_feature_positions_for_bucketing


class MultitaskBERTModel(nn.Module):
    def __init__(self, config, pretrained_weights=None):
        super(MultitaskBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)

        self.config = config  # Store config as model attribute

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
            nn.Linear(config.outcome_head_input_dim, config.hidden_dim // 2),
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

    def forward(self, input_ids, attention_mask, times, loop_features=None, customer_types=None):
        """
        Args:
            input_ids:       [batch_size, seq_len]        Token/Activity IDs
            attention_mask:  [batch_size, seq_len]        1=valid token, 0=padding
            customer_types:   [batch_size] or None         For concept drift adaptation
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

        # Truncate oversize inputs
        if input_ids.size(1) > 512:
            input_ids = input_ids[:, :512].long()
            attention_mask = attention_mask[:, :512]
            times = times[:, :512]

        # Forward through prompted BERT
        last_hidden_state = self.prompted_bert(
            input_ids,
            attention_mask=attention_mask,
            times=times,
            customer_type=customer_types
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
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp_min(1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Convert to hidden dimension
        pooled_output = self.shared_hidden(pooled_output)
        pooled_output = self.shared_activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # Concatenate loop features if provided
        # Before concatenation, ensure both tensors have same dimensions
        if loop_features is not None and loop_features.shape[1] > 0:
            # Ensure loop_features has shape [batch_size, feature_dim]
            if loop_features.dim() == 1:
                loop_features = loop_features.unsqueeze(0)
            pooled_output = torch.cat([pooled_output, loop_features], dim=1)

        outcome_logits = self.outcome_head(pooled_output)
        # Shape => [batch_size, config.num_outcomes]

        # Check for NaN in next activity logits
        if torch.isnan(next_activity_logits).any():
            print("NaN detected in next activity logits!")
            # Add debugging information about the input that caused NaN

        return next_activity_logits, outcome_logits


def compute_class_weights(dataloader, device, num_activities, num_outcomes, beta=0.999):
    """
    Compute class weights using effective samples following paper:
    "Class-Balanced Loss Based on Effective Number of Samples"
    """

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

    # Compute effective numbers
    activity_weights = (1 - beta) / (1 - beta ** (activity_counts + 1))
    outcome_weights = (1 - beta) / (1 - beta ** (outcome_counts + 1))

    # Normalize weights
    activity_weights = activity_weights / activity_weights.sum() * num_activities
    outcome_weights = outcome_weights / outcome_weights.sum() * num_outcomes

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


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target):
        # Get cross entropy loss
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        # Compute focal loss
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def train_model(model, dataloader, optimizer, device, config, num_epochs=5, print_batch_data=False,
                accumulation_steps=16):
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

    # # Separate loss functions for each task with computed weights
    # next_activity_loss_fn = nn.CrossEntropyLoss(weight=activity_weights)
    # outcome_loss_fn = nn.CrossEntropyLoss(weight=outcome_weights)

    # Create loss functions with focal loss
    # next_activity_loss_fn = FocalLoss(weight=activity_weights, gamma=2.0).to(device)
    # outcome_loss_fn = FocalLoss(weight=outcome_weights, gamma=2.0).to(device)

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
    next_activity_loss_fn = nn.CrossEntropyLoss(weight=activity_weights).to(device)
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

            input_ids = batch['trace'].long().to(device)
            attention_mask = batch['mask'].to(device)
            customer_type = batch['customer_type'].to(device)
            loop_feats = batch.get('loop_features', None)  # Use get() with default None
            times = batch['times'].to(device)
            next_activity_labels = batch['next_activity'].to(device)
            outcome_labels = batch['outcome'].to(device)

            if print_batch_data and epoch == 0 and step == 0:
                print("=== First Batch Data ===")
                print("Input IDs (trace):", input_ids)
                print("Relative Time:", times)
                print("Attention Mask:", attention_mask)
                print("Customer Types:", customer_type)
                print("Outcome Labels:", outcome_labels)
                print("========================")

            # Forward pass
            next_activity_logits, outcome_logits = model(
                input_ids,
                attention_mask,
                times,
                loop_features=loop_feats.to(device) if loop_feats is not None else None,
                customer_types=customer_type
            )

            # --- Next Activity Loss ---
            # Flatten (batch_size * seq_len) to align with cross-entropy:
            next_activity_logits_2d = next_activity_logits.view(-1, config.num_activities)
            next_activity_labels_1d = next_activity_labels.view(-1)

            # # We only want valid positions (not padded). Flatten the attention_mask:
            # valid_mask = attention_mask.view(-1).bool()
            # # Add padding mask
            # padding_mask = (input_ids != 0).view(-1).bool()
            # # Combine masks to only consider non-padding valid positions
            # final_mask = valid_mask & padding_mask
            # Get actual sequence length from model outputs
            seq_len = next_activity_logits.shape[1]  # Get the actual sequence length from model

            # Rebuild masks to match model's output sequence length
            # Truncate masks to match the model's output sequence length
            valid_mask = batch['mask'][:, :seq_len].contiguous().view(-1).bool()
            padding_mask = (batch['trace'][:, :seq_len] != 0).contiguous().view(-1).bool()
            final_mask = valid_mask & padding_mask

            # Truncate and flatten labels to match model output
            next_activity_labels = batch['next_activity'][:, :seq_len].contiguous().view(-1)  # NEW
            next_activity_labels_1d = next_activity_labels.to(device)  # MODIFIED

            # Existing code remains...
            valid_activity_logits = next_activity_logits_2d[final_mask]
            valid_activity_labels = next_activity_labels_1d[final_mask]  # Now aligned

            next_activity_loss = next_activity_loss_fn(valid_activity_logits, valid_activity_labels)
            print(f"Model output seq length: {seq_len}")
            print(f"Mask shape: {final_mask.shape}")
            print(f"Logits shape: {next_activity_logits_2d.shape}")
            print(f"Labels shape: {next_activity_labels.shape}")
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
        # Save losses to file
        with open('training_losses.csv', 'a') as f:
            if epoch == 0:  # Write header if first epoch
                f.write('epoch,total_loss,next_activity_loss,outcome_loss\n')
            f.write(f'{epoch + 1},{avg_loss:.4f},{avg_next_activity_loss:.4f},{avg_outcome_loss:.4f}\n')


def train_model_with_buckets(model, log, dataset, mappings, optimizer, device, config, num_epochs=5):
    # Get buckets using existing feature engineering
    buckets = analyze_feature_positions_for_bucketing(
        dataset,
        log,
        mappings
    )

    print("\n=== Training Buckets ===")
    for bucket_name, (min_len, max_len) in buckets.items():
        print(f"Bucket {bucket_name}: {min_len} to {max_len if max_len else 'end'}")

    # Create separate dataloaders for each bucket
    bucket_dataloaders = {}
    for bucket_name, (min_len, max_len) in buckets.items():
        # Filter dataset by trace length
        indices = []
        for i in range(len(dataset)):
            trace_length = (dataset.traces[i] != 0).sum().item()
            if min_len <= trace_length and (max_len is None or trace_length <= max_len):
                indices.append(i)

        if indices:
            bucket_dataset = torch.utils.data.Subset(dataset, indices)
            bucket_dataloaders[bucket_name] = DataLoader(
                bucket_dataset,
                batch_size=8,
                shuffle=True
            )
            print(f"Bucket {bucket_name}: {len(indices)} samples")

    # Training loop with buckets
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for bucket_name, dataloader in bucket_dataloaders.items():
            print(f"\nTraining on {bucket_name} traces...")
            train_model(model, dataloader, optimizer, device, config, num_epochs=1)


def train_model_with_curriculum(model, dataset, mappings, optimizer, device, config, num_epochs=5):
    """Train model using curriculum learning with progressive difficulty and balanced sampling"""

    # Get buckets using trace length distribution
    trace_lengths = []
    for i in range(len(dataset)):
        trace = dataset.traces[i]
        trace_length = (trace != 0).sum().item()
        trace_lengths.append(trace_length)

    # Calculate length percentiles for balanced bucketing
    length_25th = np.percentile(trace_lengths, 25)
    length_75th = np.percentile(trace_lengths, 75)

    buckets = {
        'short': (1, int(length_25th)),
        'medium': (int(length_25th) + 1, int(length_75th)),
        'long': (int(length_75th) + 1, None)
    }

    print("\n=== Curriculum Learning Stages ===")
    for stage, (bucket_name, (min_len, max_len)) in enumerate(buckets.items()):
        print(f"Stage {stage + 1}: {bucket_name} (length {min_len} to {max_len if max_len else 'end'})")

    stage_metrics = {}
    accumulated_indices = []

    # Train progressively through stages
    for stage, (bucket_name, (min_len, max_len)) in enumerate(buckets.items()):
        print(f"\n=== Training Stage {stage + 1}: {bucket_name} ===")

        # Get indices for current stage
        stage_indices = []
        for i in range(len(dataset)):
            trace_length = (dataset.traces[i] != 0).sum().item()
            if min_len <= trace_length <= (max_len if max_len else float('inf')):
                stage_indices.append(i)

        # Add current stage indices to accumulated indices
        accumulated_indices.extend(stage_indices)

        # Create balanced dataset for current curriculum stage
        stage_dataset = Subset(dataset, accumulated_indices)
        stage_dataloader = DataLoader(
            stage_dataset,
            batch_size=8,
            shuffle=True
        )

        print(f"Samples in current stage: {len(accumulated_indices)}")

        # Train for this stage
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs} (Stage {stage + 1})")

            # Compute class weights for current stage
            activity_weights, outcome_weights = compute_class_weights(
                stage_dataloader, device, config.num_activities, config.num_outcomes
            )

            # Initialize loss functions with class weights
            next_activity_loss_fn = FocalLoss(
                weight=activity_weights,
                gamma=2.0
            ).to(device)

            outcome_loss_fn = FocalLoss(
                weight=outcome_weights,
                gamma=2.0
            ).to(device)

            # Training loop
            model.train()
            total_loss = 0
            next_activity_total_loss = 0
            outcome_total_loss = 0

            progress_bar = tqdm(enumerate(stage_dataloader),
                                total=len(stage_dataloader),
                                desc=f"Stage {stage + 1} Epoch {epoch + 1}")

            for step, batch in progress_bar:
                optimizer.zero_grad()

                # Forward pass
                next_activity_logits, outcome_logits = model(
                    batch['trace'].to(device),
                    batch['mask'].to(device),
                    batch['times'].to(device),
                    batch['loop_features'].to(device),
                    batch['customer_type'].to(device)
                )

                # Calculate next activity loss with masking
                valid_mask = batch['mask'].view(-1).bool()
                padding_mask = (batch['trace'] != 0).view(-1).bool()
                final_mask = valid_mask & padding_mask

                next_act_logits_2d = next_activity_logits.view(-1, config.num_activities)
                next_act_labels_1d = batch['next_activity'].view(-1).to(device)

                valid_logits = next_act_logits_2d[final_mask]
                valid_labels = next_act_labels_1d[final_mask]

                next_activity_loss = next_activity_loss_fn(valid_logits, valid_labels)

                # Calculate outcome loss
                outcome_loss = outcome_loss_fn(
                    outcome_logits,
                    batch['outcome'].to(device)
                )

                # Combined loss with dynamic weighting
                loss = 0.6 * next_activity_loss + 0.4 * outcome_loss

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                next_activity_total_loss += next_activity_loss.item()
                outcome_total_loss += outcome_loss.item()

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'next_act': f"{next_activity_loss.item():.4f}",
                    'outcome': f"{outcome_loss.item():.4f}"
                })

            # Store stage metrics
            stage_metrics[f"stage_{stage + 1}_epoch_{epoch + 1}"] = {
                'avg_loss': total_loss / len(stage_dataloader),
                'next_activity_loss': next_activity_total_loss / len(stage_dataloader),
                'outcome_loss': outcome_total_loss / len(stage_dataloader)
            }

    return stage_metrics

