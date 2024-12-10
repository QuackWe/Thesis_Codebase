import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
from torch.optim import AdamW
from tqdm import tqdm
import os
from sys import argv

log = argv[1]


class MaskedActivityDataset(Dataset):
    def __init__(self, data_file, tokenizer, mask_prob=0.15):
        """
        Dataset for Masked Activity Modeling (MAM).
        :param data_file: Path to the preprocessed data file.
        :param tokenizer: Hugging Face BERT tokenizer.
        :param mask_prob: Probability of masking each activity.
        """
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

        # Compute max length dynamically based on the longest trace
        self.max_len = self.get_max_trace_length()

    def get_max_trace_length(self):
        """Compute the length of the longest trace."""
        trace_lengths = [len(eval(trace)) for trace in self.data["Prefix"]]
        return max(trace_lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract prefix and apply random masking
        original_activities = eval(self.data.iloc[idx]["Prefix"])  # Convert string to list
        masked_activities, labels = self.apply_random_masking(original_activities)

        input_ids = []
        label_ids = []

        for activity, label in zip(masked_activities, labels):
            if label != '[PAD]':
                # Masked activity
                orig_tokens = self.tokenizer.tokenize(label)
                orig_token_ids = self.tokenizer.convert_tokens_to_ids(orig_tokens)
                input_ids.extend([self.tokenizer.mask_token_id] * len(orig_token_ids))
                label_ids.extend(orig_token_ids)
            else:
                # Unmasked activity
                tokens = self.tokenizer.tokenize(activity)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_ids.extend(token_ids)
                label_ids.extend([0] * len(token_ids))

        # Pad sequences
        input_ids = input_ids[:self.max_len]
        label_ids = label_ids[:self.max_len]

        padding_length = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        label_ids += [0] * padding_length
        attention_mask = [1] * len(input_ids)
        if padding_length > 0:
            attention_mask[-padding_length:] = [0] * padding_length

        # Generate position_ids
        position_ids = torch.arange(self.max_len, dtype=torch.long)  # Shape: [max_len]

        # Debugging: Validate alignment
        for i, token_id in enumerate(input_ids):
            if token_id == self.tokenizer.mask_token_id:
                assert label_ids[i] != 0, f"Mismatch at position {i}: Masked token but no label."
            else:
                assert label_ids[i] == 0, f"Mismatch at position {i}: Unmasked token with label."

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "position_ids": position_ids,

        }

    def apply_random_masking(self, activities, strategy="80-10-10"):
        """
        Apply random masking to the activities while ensuring alignment between masking and labeling.
        """
        masked_activities = activities[:]
        labels = ["[PAD]"] * len(activities)  # Default all labels to [PAD]

        # Handle short traces: always mask at least one activity
        if len(activities) == 1:
            masked_activities[0] = "[MASK]"
            labels[0] = activities[0]
            return masked_activities, labels

        if strategy == "uniform":
            num_to_mask = max(1, int(len(activities) * self.mask_prob))
            masked_indices = torch.randperm(len(activities))[:num_to_mask]
        elif strategy == "80-10-10":
            num_to_mask = max(1, int(len(activities) * self.mask_prob))
            masked_indices = torch.randperm(len(activities))[:num_to_mask]
            for idx in masked_indices:
                if torch.rand(1).item() < 0.8:
                    masked_activities[idx] = "[MASK]"
                    labels[idx] = activities[idx]
                elif torch.rand(1).item() < 0.5:
                    masked_activities[idx] = random.choice(activities)  # Replace with random activity
                else:
                    labels[idx] = "[PAD]"  # Leave as-is
        else:
            raise ValueError("Invalid masking strategy.")

        # Debugging: Validate alignment between masking and labels
        for i, token in enumerate(masked_activities):
            if labels[i] != "[PAD]":  # If label is set, token must be [MASK]
                assert token == "[MASK]", f"[ERROR] Label mismatch at index {i}: Token={token}, Label={labels[i]}"

        return masked_activities, labels

# Define paths and tokenizer
data_file = "datasets/"+log+"/preprocessed_prefixes.csv"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create the dataset with random masking
masked_dataset = MaskedActivityDataset(data_file, tokenizer, mask_prob=0.15)

dataloader = DataLoader(masked_dataset, batch_size=16, shuffle=True)

# Define the MAM training loop
def train_mam(dataloader, model, optimizer, device, num_epochs=3, accumulation_steps=2):
    """
    Train the BERT model for Masked Activity Modeling (MAM).
    :param dataloader: PyTorch DataLoader for training data.
    :param model: BERT model for masked language modeling.
    :param optimizer: Optimizer for training.
    :param device: Device to train on (CPU or GPU).
    :param num_epochs: Number of training epochs.
    """
    # Set the model to training mode
    model.train()

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

    scaler = GradScaler()  # Initialize GradScaler for mixed precision training

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            position_ids = batch["position_ids"].to(device)

            with autocast():  # Enable mixed precision
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids,
                )
                loss = outputs.loss / accumulation_steps

            # Scale the loss for mixed precision
            scaler.scale(loss).backward()

            # Backward pass
            # optimizer.zero_grad()
            if (i + 1) % accumulation_steps == 0 or (i +1) == len(dataloader):
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        print(f"Epoch {epoch+1}: Average Loss = {total_loss / len(dataloader)}")

torch.set_printoptions(threshold=10000, edgeitems=10, linewidth=1000)

# Set up training components
def setup_training(data_file, batch_size=16, learning_rate=5e-5, num_epochs=3, print_examples=False, save_model=True, save_path='saved_model'):
    """
    Prepare the data, model, optimizer, and device for MAM training.
    :param data_file: Path to the preprocessed dataset.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for AdamW optimizer.
    :param num_epochs: Number of training epochs.
    :param print_examples: Whether to print example traces and their masked versions.
    :param save_model: Whether to save the model after training.
    :param save_path: Path to save the trained model.
    """
    # Load tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MaskedActivityDataset(data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    model.to(device)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    train_mam(dataloader, model, optimizer, device, num_epochs=num_epochs)

    # Save the model after training
    if save_model:
        print(f"Saving model to '{save_path}'")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

# Start training
setup_training(data_file, batch_size=16, learning_rate=5e-5, num_epochs=10, save_model=True, save_path='mam_pretrained_model')
