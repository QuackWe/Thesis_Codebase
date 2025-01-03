import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class MultitaskBERTModel(nn.Module):
    def __init__(self, config, pretrained_weights=None):
        super(MultitaskBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)

        # Freeze the first half of BERT layers to stabilize training
        # BERT-base has 12 layers, freeze first 6
        for name, param in self.bert.named_parameters():
            if "encoder.layer." in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < 6:
                    param.requires_grad = False

        self.dropout = nn.Dropout(p=0.1)
        self.shared_hidden = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.shared_activation = nn.ReLU()

        # Next-Activity head (per step)
        # Instead of average pooling, we directly map from [batch, seq_len, hidden_dim] -> [batch, seq_len, num_activities]
        self.next_activity_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),  # 768 -> (768)
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_activities)
        )

        # Outcome head (single label per sequence)
        # We'll do average pooling or final-step for outcome. Let’s do average for now.
        self.outcome_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_outcomes)
        )

    def forward(self, input_ids, attention_mask, customer_type=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # (Optional) apply dropout to each time step.
        seq_output = self.dropout(last_hidden_state)  # [batch_size, seq_len, hidden_dim]

        # For next-activity: produce [batch_size, seq_len, num_activities]
        next_activity_logits = self.next_activity_head(seq_output)

        # Outcome: Single vector per sequence
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask  # shape [batch, hidden_dim]

        pooled_output = self.shared_activation(self.shared_hidden(pooled_output))
        pooled_output = self.dropout(pooled_output)
        outcome_logits = self.outcome_head(pooled_output)
        # shape: [batch, num_outcomes]

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


def train_model(model, dataloader, optimizer, device, config, num_epochs=5, print_batch_data=False,
                accumulation_steps=2):
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
    next_activity_loss_fn = nn.CrossEntropyLoss()
    # Weighted CE for outcome prediction
    outcome_loss_fn = nn.CrossEntropyLoss(weight=outcome_weights)

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
            input_ids = batch['trace'].to(device)
            attention_mask = batch['mask'].to(device)
            customer_types = batch['customer_type'].to(device)
            next_activity_labels = batch['next_activity'].to(device)
            outcome_labels = batch['outcome'].to(device)

            if print_batch_data and epoch == 0 and step == 0:
                print("=== First Batch Data ===")
                print("Input IDs (trace):", input_ids)
                print("Attention Mask:", attention_mask)
                print("Customer Types:", customer_types)
                print("Outcome Labels:", outcome_labels)
                print("========================")

            # Forward pass
            next_activity_logits, outcome_logits = model(input_ids, attention_mask, customer_types)

            # --- Next Activity Loss ---
            # Flatten (batch_size * seq_len) to align with cross-entropy:
            next_activity_logits_2d = next_activity_logits.view(-1, config.num_activities)
            next_activity_labels_1d = next_activity_labels.view(-1)

            # We only want valid positions (not padded). Flatten the attention_mask:
            valid_mask = attention_mask.view(-1).bool()
            valid_activity_logits = next_activity_logits_2d[valid_mask]
            valid_activity_labels = next_activity_labels_1d[valid_mask]

            next_activity_loss = next_activity_loss_fn(valid_activity_logits, valid_activity_labels)

            # --- Outcome Loss ---
            # One outcome per sequence, so no flattening.
            # outcome_logits: [batch_size, num_outcomes], outcome_labels: [batch_size]
            outcome_loss = outcome_loss_fn(outcome_logits, outcome_labels)

            # Combine losses with task weights
            # You might want to adjust these weights based on task importance
            task1_weight = 0.5  # Next activity prediction
            task2_weight = 0.5  # Outcome prediction
            loss = (task1_weight * next_activity_loss +
                    task2_weight * outcome_loss) / accumulation_steps

            # Backward pass
            loss.backward()
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