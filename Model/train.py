from model import MultitaskBERTModel, train_model
from dataloader import TraceDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchinfo import summary
from transformers import BertTokenizer, BertForMaskedLM
from sys import argv
import os
from mam import train_mam, MaskedActivityDataset  # Import the MAM training function


# Configuration setup
class Config:
    def __init__(self, pytorch_dataset):
        unique_activities = pytorch_dataset.traces.max().item() + 1  # Assuming activity IDs are zero-indexed
        self.num_activities = unique_activities
        self.max_seq_length = pytorch_dataset.traces.shape[1]  # Second dimension of traces tensor
        self.num_outcomes = pytorch_dataset.outcomes.max().item() + 1  # Assuming outcomes are zero-indexed
        self.embedding_dim = 768  # BERT-base hidden size
        self.time_embedding_dim = self.embedding_dim  # Use same dimension as activities
        self.hidden_dim = self.embedding_dim
        self.bert_model = "bert-base-uncased"  # Pretrained BERT model
        self.num_heads = 12  # Standard number of attention heads in BERT-base
        self.g_prompt_length = 10
        self.e_prompt_length = 10
        self.prompt_prefix_size = 5
        self.prefix_tune = True


# Main block for training
if __name__ == "__main__":
    log = argv[1]

    # Load the saved dataset
    pytorch_dataset = torch.load(f'datasets/{log}/pytorch_dataset_consistent.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    # Check if MAM is already pretrained
    pretrained_weights = f"datasets/{log}/mam_pretrained_model"
    if not os.path.exists(pretrained_weights):
        # If MAM not pretrained, do the pretraining
        print("Starting MAM pretraining...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        traces = [trace.tolist() for trace in pytorch_dataset.traces]  # Extract traces
        times = [time.tolist() for time in pytorch_dataset.times]  # Extract times

        # Create MAM dataset with temporal features
        mam_dataset = MaskedActivityDataset(traces, times, tokenizer)
        mam_dataloader = DataLoader(mam_dataset, batch_size=16, shuffle=True)

        mam_model = BertForMaskedLM.from_pretrained("bert-base-uncased")  # Initialize BERT for MAM
        mam_model.to(device)

        mam_optimizer = torch.optim.AdamW(mam_model.parameters(), lr=5e-5)
        train_mam(mam_dataloader, mam_model, mam_optimizer, device, num_epochs=10)  # Pretrain the model with MAM

        # Save the pretrained model
        mam_model.save_pretrained(pretrained_weights)
        print(f"MAM pretraining complete. Model saved to '{pretrained_weights}'")
    else:
        print("MAM already pretrained. Skipping MAM pretraining.")

    # Create configuration for fine-tuning
    config = Config(pytorch_dataset)

    # Create DataLoader for fine-tuning
    dataloader = DataLoader(pytorch_dataset, batch_size=8, shuffle=True)  # Adjust batch size as needed

    # Initialize Multitask Model with pretrained weights
    multitask_model = MultitaskBERTModel(config, pretrained_weights=pretrained_weights).to(device)
    optimizer = torch.optim.Adam(multitask_model.parameters(), lr=1e-4)

    # Fine-tune the Multitask Model
    print("Starting multitask fine-tuning...")
    train_model(multitask_model, dataloader, optimizer, device, config, num_epochs=10)

    # Save the fine-tuned model
    multitask_model.prompted_bert.e_prompt.save_prompts(f'datasets/{log}')
    save_path = f'datasets/{log}/multitask_bert_model.pth'
    torch.save(multitask_model.state_dict(), save_path)
    print(f"Fine-tuned multitask model saved to '{save_path}'")
