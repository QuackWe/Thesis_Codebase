import torch.nn as nn
import torch

# Transformer Block for MTLFormer
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention
        attn_output, _ = self.attention(x, x, x)
        # Add & Norm
        x = self.norm1(attn_output + x)
        # Feed Forward Network
        ffn_output = self.ffn(x)
        # Add & Norm
        out = self.norm2(ffn_output + x)
        return out


# Multi-task Learning Transformer model (MTLFormer)
class MTLFormer(nn.Module):
    def __init__(self, embed_size, heads, dropout, num_classes):
        super(MTLFormer, self).__init__()
        # Embedding layer
        self.transformer = TransformerBlock(embed_size, heads, dropout)
        # Task-specific output heads
        self.fc_activity = nn.Linear(embed_size, num_classes)  # Next Activity (classification)
        self.fc_time = nn.Linear(embed_size, 1)  # Next Event Time (regression)
        self.fc_remaining = nn.Linear(embed_size, 1)  # Remaining Time (regression)

    def forward(self, x):
        # Pass through transformer block
        x = self.transformer(x)
        # Task-specific outputs
        next_activity = self.fc_activity(x)
        next_time = self.fc_time(x)
        remaining_time = self.fc_remaining(x)
        return next_activity, next_time, remaining_time

# Loss functions
def multitask_loss(activity_pred, time_pred, remaining_pred, activity_label, time_label, remaining_label, weights):
    # Task A: Cross Entropy Loss for Next Activity Prediction (classification)
    activity_loss = nn.CrossEntropyLoss()(activity_pred, activity_label)

    # Task B: Mean Absolute Error (MAE) for Next Event Time Prediction (regression)
    time_loss = nn.L1Loss()(time_pred, time_label)

    # Task C: MAE for Remaining Time Prediction (regression)
    remaining_loss = nn.L1Loss()(remaining_pred, remaining_label)

    # Total loss = Weighted sum of losses
    total_loss = weights[0] * activity_loss + weights[1] * time_loss + weights[2] * remaining_loss
    return total_loss


# Training loop
def train_model(model, dataloader, optimizer, weights, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            # Get data for each task
            sequences = batch['sequence']  # Activity sequences (input features)
            next_activity_labels = batch['next_activity']  # Next activity (classification labels)
            next_event_time_labels = batch['next_event_time']  # Next event time (regression labels)
            remaining_time_labels = batch['remaining_time']  # Remaining time (regression labels)

            # Forward pass
            activity_pred, time_pred, remaining_pred = model(sequences)

            # Squeeze the prediction to remove the extra dimension
            time_pred = torch.squeeze(time_pred)  # For next_event_time prediction
            remaining_pred = torch.squeeze(remaining_pred)  # For remaining_time prediction

            # Compute loss
            loss = multitask_loss(activity_pred, time_pred, remaining_pred,
                                  next_activity_labels, next_event_time_labels, remaining_time_labels, weights)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    return epoch, num_epochs, total_loss