from model import MultitaskBERTModel, train_model, train_model_with_buckets, train_model_with_curriculum
from dataloader import TraceDataset
from FeatureEngineering import add_all_features, loop_activities_by_outcome, time_sensitive_transitions
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchinfo import summary
from transformers import BertTokenizer, BertForMaskedLM
from sys import argv
import os
from mam import train_mam, MaskedActivityDataset  # Import the MAM training function
from preprocess import MemoryMappedDataset
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()


# Configuration setup
class Config:
    def __init__(self, pytorch_dataset):
        if hasattr(pytorch_dataset, 'trace_mmap'):  # If it's a memory-mapped dataset
            # Get num_activities from trace data
            self.num_activities = int(pytorch_dataset.trace_mmap.max()) + 1
            self.max_seq_length = pytorch_dataset.trace_mmap.shape[1]
            # Get num_outcomes from metadata
            self.num_outcomes = max(pytorch_dataset.meta['Outcome']) + 1
        else:  # Regular dataset
            self.num_activities = pytorch_dataset.traces.max().item() + 1
            self.max_seq_length = pytorch_dataset.traces.shape[1]
            self.num_outcomes = pytorch_dataset.outcomes.max().item() + 1

        # Rest of the configuration remains the same
        self.embedding_dim = 768
        self.time_embedding_dim = self.embedding_dim
        self.hidden_dim = self.embedding_dim
        self.bert_model = "bert-base-uncased"
        self.num_heads = 12
        self.g_prompt_length = 10
        self.e_prompt_length = 10
        self.prompt_prefix_size = 5
        self.prefix_tune = True
        self.loop_feat_dim = sum([len(activities) for activities in loop_activities_by_outcome.values()])
        self.temporal_feat_dim = len(time_sensitive_transitions) * 2
        self.total_feature_dim = self.loop_feat_dim + self.temporal_feat_dim
        self.outcome_head_input_dim = self.embedding_dim + self.total_feature_dim


# Main block for training
if __name__ == "__main__":
    log = argv[1]

    # Load the saved dataset
    pytorch_data_ref = torch.load(f'datasets/{log}/pytorch_dataset_consistent.pt')

    if isinstance(pytorch_data_ref, dict) and 'dataset_dir' in pytorch_data_ref:
        # It's the big dataset with memory-mapped .dat files
        dataset_info = pytorch_data_ref
        pytorch_dataset = MemoryMappedDataset(dataset_info['dataset_dir'])
    else:
        # It’s the smaller or older dataset that’s already a TraceDataset
        pytorch_dataset = pytorch_data_ref

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    # Load mappings
    mappings = torch.load(f'datasets/{log}/mappings_consistent.pt')

    # Add both loop and temporal features
    add_all_features(pytorch_dataset, mappings, loop_activities_by_outcome,
                     time_sensitive_transitions)

    # # Update collate_fn to include loop features
    # def collate_fn(batch):
    #     collated_batch = {
    #         'trace': torch.stack([item['trace'] for item in batch]),
    #         'times': torch.stack([item['times'] for item in batch]),
    #         'next_activity': torch.stack([item['next_activity'] for item in batch]),
    #         'mask': torch.stack([item['mask'] for item in batch]),
    #         'outcome': torch.stack([item['outcome'] for item in batch]),
    #         'customer_type': torch.stack([item['customer_type'] for item in batch]),
    #         'loop_features': torch.stack([item['loop_features'] for item in batch])
    #     }
    #     return collated_batch

    # Create DataLoader with updated collate_fn
    dataloader = DataLoader(pytorch_dataset, batch_size=8, shuffle=True)

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
        mam_dataloader = DataLoader(mam_dataset, batch_size=4, shuffle=True)

        mam_model = BertForMaskedLM.from_pretrained(
            "bert-base-uncased",
            # attention_probs_dropout_prob=0.1,
            # hidden_dropout_prob=0.1,
            use_cache=False  # Disable key/value caching
        )  # Initialize BERT for MAM

        mam_model.gradient_checkpointing_enable()
        mam_model.to(device)

        mam_optimizer = torch.optim.AdamW(mam_model.parameters(), lr=5e-5)
        train_mam(mam_dataloader, mam_model, mam_optimizer, device, num_epochs=50)  # Pretrain the model with MAM

        # Save the pretrained model
        mam_model.save_pretrained(pretrained_weights)
        print(f"MAM pretraining complete. Model saved to '{pretrained_weights}'")
    else:
        print("MAM already pretrained. Skipping MAM pretraining.")

    # Create configuration for fine-tuning
    config = Config(pytorch_dataset)

    # Create DataLoader for fine-tuning
    # dataloader = DataLoader(pytorch_dataset, batch_size=8, shuffle=True)  # Adjust batch size as needed

    # Initialize Multitask Model with pretrained weights
    multitask_model = MultitaskBERTModel(config, pretrained_weights=pretrained_weights).to(device)
    optimizer = torch.optim.Adam(multitask_model.parameters(), lr=1e-4)

    # Fine-tune the Multitask Model
    print("Starting multitask fine-tuning...")
    # train_model(multitask_model, dataloader, optimizer, device, config, num_epochs=15)
    # train_model_with_buckets(multitask_model, pytorch_dataset, mappings, optimizer, device, config, num_epochs=1)
    stage_metrics = train_model_with_curriculum(
        multitask_model,
        pytorch_dataset,
        mappings,
        optimizer,
        device,
        config,
        num_epochs=15
    )
    # Save the fine-tuned model
    multitask_model.prompted_bert.e_prompt.save_prompts(f'datasets/{log}')
    save_path = f'datasets/{log}/multitask_bert_model.pth'
    torch.save(multitask_model.state_dict(), save_path)
    print(f"Fine-tuned multitask model saved to '{save_path}'")
