import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from tqdm import tqdm

# class MultitaskBERTModel(nn.Module):
#     def __init__(self, config, pretrained_weights=None):
#         super(MultitaskBERTModel, self).__init__()
#         # Load pretrained weights if available
#         # self.bert = BertModel.from_pretrained(pretrained_weights) if pretrained_weights else BertModel(config)
#         self.bert = BertModel.from_pretrained(pretrained_weights)
#         self.next_activity_head = nn.Linear(config.embedding_dim, config.num_activities)
#         self.outcome_head = nn.Linear(config.embedding_dim, config.num_outcomes)

#     def forward(self, input_ids, attention_mask, return_probs=False):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         sequence_output = outputs.last_hidden_state
#         cls_output = outputs.pooler_output

#         # Task-specific predictions
#         next_activity_logits = self.next_activity_head(sequence_output[:, -1, :])
#         outcome_logits = self.outcome_head(cls_output)

#         if return_probs:
#             # Apply softmax to logits to return probabilities
#             next_activity_probs = F.softmax(next_activity_logits, dim=-1)
#             outcome_probs = F.softmax(outcome_logits, dim=-1)
#             return next_activity_probs, outcome_probs

#         return next_activity_logits, outcome_logits

class MultitaskBERTModel(nn.Module):
    def __init__(self, config, pretrained_weights=None):
        super(MultitaskBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_weights)

        # Shared layers
        self.shared_hidden = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.shared_activation = nn.ReLU()

        # Task-specific layers
        self.next_activity_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_activities)
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_outcomes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # [CLS] token's output

        # Shared representation
        shared_output = self.shared_activation(self.shared_hidden(cls_output))

        # Task-specific predictions
        next_activity_logits = self.next_activity_head(shared_output)
        outcome_logits = self.outcome_head(shared_output)

        return next_activity_logits, outcome_logits

def train_model(model, dataloader, optimizer, loss_fn, device, num_epochs=5, print_batch_data=False, accumulation_steps=2):
    """
    Fine-tune the multitask model.
    Args:
        model: MultitaskBERTModel instance.
        dataloader: PyTorch DataLoader containing the fine-tuning dataset.
        optimizer: Optimizer for training.
        loss_fn: Loss function for training.
        device: Training device (CPU or GPU).
        num_epochs: Number of training epochs.
        accumulation_steps: Number of batches to accumulate gradients before an optimization step.

    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

        optimizer.zero_grad()  # Reset gradients before accumulation

        for step, batch in progress_bar:
            input_ids = batch['trace'].to(device)
            attention_mask = batch['mask'].to(device)
            next_activity_labels = batch['outcome'].to(device)
            customer_types = batch['customer_type'].to(device)
            
            # Optional: Print data for the first batch of the first epoch
            if print_batch_data and epoch == 0 and step == 0:
                print("=== First Batch Data ===")
                print("Input IDs (trace):", input_ids)
                print("Attention Mask:", attention_mask)
                print("Next Activity Labels (outcome):", next_activity_labels)
                print("Customer Types:", customer_types)
                print("========================")

            # Forward pass
            next_activity_logits, outcome_logits = model(input_ids, attention_mask)

            # Compute loss and scale by accumulation steps
            next_activity_loss = loss_fn(next_activity_logits, next_activity_labels)
            outcome_loss = loss_fn(outcome_logits, customer_types)
            # Weighted loss
            task1_loss_weight = 0.7  # Adjust based on task importance
            task2_loss_weight = 0.3
            loss = (task1_loss_weight * next_activity_loss) + ( task2_loss_weight * outcome_loss) / accumulation_steps

            # Backward pass
            loss.backward()

            # Perform optimizer step every `accumulation_steps` batches
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()  # Reset gradients after optimization step

            total_loss += loss.item() * accumulation_steps  # Scale back to original loss

        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {total_loss / len(dataloader):.4f}")

            

            