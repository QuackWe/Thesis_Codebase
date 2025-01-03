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


class MaskedActivityDataset(Dataset):
    def __init__(self, traces, tokenizer, mask_prob=0.15):
        """
        Dataset for Masked Activity Modeling (MAM).
        :param traces: List of activity sequences (list of lists).
        :param tokenizer: Hugging Face BERT tokenizer.
        :param mask_prob: Probability of masking each activity.
        :param max_seq_length: Maximum length of activity traces.
        """
        self.traces = traces
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.max_seq_length = self.get_max_trace_length()  # Dynamically compute max sequence length

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        # Original trace
        original_trace = self.traces[idx]

        # Apply random masking
        masked_trace, labels = self.apply_random_masking(original_trace)

        # Tokenize trace
        tokenized_trace = self.tokenizer(
            masked_trace,
            is_split_into_words=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tokenized_labels = self.tokenizer(
            labels,
            is_split_into_words=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized_trace["input_ids"].squeeze(0)  # Shape: (max_seq_length)
        attention_mask = tokenized_trace["attention_mask"].squeeze(0)
        labels = tokenized_labels["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def get_max_trace_length(self):
        """Compute the length of the longest trace."""
        return max(len(trace) for trace in self.traces)

    def apply_random_masking(self, activities):
        """Apply random masking to the activities."""
        activities = [str(a) for a in activities]
        masked_activities = activities[:]
        labels = ["[PAD]"] * len(activities)

        num_to_mask = max(1, int(len(activities) * self.mask_prob))
        masked_indices = random.sample(range(len(activities)), num_to_mask)

        for idx in masked_indices:
            if random.random() < 0.8:
                masked_activities[idx] = "[MASK]"
                labels[idx] = activities[idx]
            elif random.random() < 0.5:
                masked_activities[idx] = random.choice(activities)
            else:
                labels[idx] = "[PAD]"

        return masked_activities, labels


def train_mam(dataloader, model, optimizer, device, num_epochs=3, accumulation_steps=2):
    """
    Train the BERT model for Masked Activity Modeling (MAM) with gradient accumulation.
    :param accumulation_steps: Number of batches to accumulate gradients before an optimization step.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        optimizer.zero_grad()  # Zero gradients before the accumulation loop

        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps  # Divide loss by accumulation steps
            loss.backward()  # Accumulate gradients

            total_loss += loss.item()

            # Perform optimizer step every `accumulation_steps` batches
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()  # Clear accumulated gradients

            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        print(f"Epoch {epoch + 1}: Average Loss = {total_loss / len(dataloader)}")

    # print("MAM pretraining complete. Saving model...")
    # model.save_pretrained('datasets/mam_pretrained_model')
    # print("Pretrained model saved.")
