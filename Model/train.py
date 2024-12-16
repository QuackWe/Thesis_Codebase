from model import MultitaskBERTModel, train_model
from dataloader import TraceDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchinfo import summary
from torchviz import make_dot
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
        self.hidden_dim = self.embedding_dim * 2
        self.bert_model = "bert-base-uncased"  # Pretrained BERT model

# Main block for training
if __name__ == "__main__":
    log = argv[1]

    # Load the saved dataset
    pytorch_dataset = torch.load(f'datasets/{log}/pytorch_dataset.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    
    # Check if MAM is already pretrained
    pretrained_weights = f"datasets/{log}/mam_pretrained_model"
    if not os.path.exists(pretrained_weights):
        # If MAM not pretrained, do the pretraining
        print("Starting MAM pretraining...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        traces = [trace.tolist() for trace in pytorch_dataset.traces]  # Extract traces

        mam_dataset = MaskedActivityDataset(traces, tokenizer)
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
    # Create configuration
    config = Config(pytorch_dataset)

    # Create DataLoader for fine-tuning
    dataloader = DataLoader(pytorch_dataset, batch_size=16, shuffle=True)  # Adjust batch size as needed

    # Initialize Multitask Model with pretrained weights
    multitask_model = MultitaskBERTModel(config, pretrained_weights=pretrained_weights).to(device)
    optimizer = torch.optim.Adam(multitask_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()


    # Generate a sample input for the model
    sample_input = torch.randint(0, config.num_activities, (1, config.max_seq_length)).to(device)
    sample_attention_mask = torch.ones_like(sample_input).to(device)

    summary(
    multitask_model,
    input_data=(sample_input, sample_attention_mask),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    device=device
    )

    # Perform a forward pass to get the computation graph
    outputs = multitask_model(sample_input, sample_attention_mask)

    # Generate a visualization of the model's computation graph
    dot = make_dot(outputs, params=dict(multitask_model.named_parameters()))
    # dot.render("model_architecture", format="png")  # Save as PNG, not working on tue server

    # Fine-tune the Multitask Model
    print("Starting multitask fine-tuning...")
    train_model(multitask_model, dataloader, optimizer, loss_fn, device, num_epochs=1)

    # Save the fine-tuned model
    torch.save(multitask_model.state_dict(), f'datasets/{log}/multitask_bert_model.pth')
    print(f"Fine-tuned multitask model saved to 'datasets/{log}/multitask_bert_model.pth'")
