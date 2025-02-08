import matplotlib.pyplot as plt
import numpy as np
from sys import argv
import sys
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import defaultdict
from sys import argv
sys.path.append('..')  # Add parent directory to path
from train import Config
from model import MultitaskBERTModel
from FeatureEngineering import loop_activities_by_outcome, time_sensitive_transitions
from eval import analyze_feature_positions_for_bucketing


log = argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# Load the saved dataset
mappings = torch.load(f'datasets/{log}/mappings_consistent.pt')
pytorch_dataset = torch.load('datasets/' + log + '/pytorch_dataset_consistent.pt')
pretrained_weights = f"datasets/{log}/mam_pretrained_model"

config = Config(pytorch_dataset)
model = MultitaskBERTModel(config, pretrained_weights=pretrained_weights).to(device)

# Set style for all plots
plt.style.use('seaborn')

def plot_prefix_performance(model, dataset, device):
    """Plot f1 and accuracy scores for different prefix lengths"""
    prefix_metrics = defaultdict(lambda: {'f1': [], 'accuracy': []})

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            trace_length = (example['trace'] != 0).sum().item()

            # Evaluate at different prefix lengths
            for prefix_len in range(2, trace_length + 1):
                trace = example['trace'][:prefix_len].unsqueeze(0).to(device)
                mask = example['mask'][:prefix_len].unsqueeze(0).to(device)
                times = example['times'][:prefix_len].unsqueeze(0).to(device)
                customer_type = example['customer_type'].unsqueeze(0).to(device)
                loop_features = example['loop_features'].unsqueeze(0).to(device)

                # Ensure correct feature dimension
                expected_feat_dim = model.config.total_feature_dim
                if loop_features.shape[1] != expected_feat_dim:
                    if loop_features.shape[1] < expected_feat_dim:
                        padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                        loop_features = torch.cat([loop_features, padding], dim=1)
                    else:
                        loop_features = loop_features[:, :expected_feat_dim]

                _, outcome_logits = model(trace, mask, times, loop_features, customer_type)
                pred_outcome = torch.argmax(outcome_logits, dim=1).cpu().item()
                true_outcome = example['outcome'].item()

                # Store metrics for this prefix length
                prefix_metrics[prefix_len]['true'] = true_outcome
                prefix_metrics[prefix_len]['pred'] = pred_outcome

    # Compute metrics
    prefix_lengths = sorted(prefix_metrics.keys())
    f1_scores = []
    accuracies = []

    for length in prefix_lengths:
        true = [prefix_metrics[length]['true']]
        pred = [prefix_metrics[length]['pred']]
        f1_scores.append(f1_score(true, pred, average='macro', zero_division=0))
        accuracies.append(accuracy_score(true, pred))

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(prefix_lengths, f1_scores, label='F1 Score', marker='o')
    plt.plot(prefix_lengths, accuracies, label='Accuracy', marker='s')
    plt.xlabel('Prefix Length')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Prefix Length')
    plt.legend()
    plt.grid(True)
    plt.savefig('prefix_performance.png')
    plt.close()

def plot_epoch_performance(log_file):
    """Plot performance metrics across training epochs"""
    epochs = []
    f1_scores = []
    accuracies = []

    # Parse training log file
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'completed' in line:
                epoch = int(line.split('/')[0].split()[-1])
                epochs.append(epoch)
            elif 'F1 Score:' in line:
                f1_scores.append(float(line.split(':')[1]))
            elif 'Accuracy:' in line:
                accuracies.append(float(line.split(':')[1]))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, f1_scores, label='F1 Score', marker='o')
    plt.plot(epochs, accuracies, label='Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Performance Metrics Across Training Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('epoch_performance.png')
    plt.close()

def plot_bucket_performance(model, dataset, device):
    """Plot performance metrics for different trace length buckets"""
    buckets = analyze_feature_positions_for_bucketing(
        dataset,
        mappings,
        loop_activities_by_outcome,
        time_sensitive_transitions
    )

    bucket_metrics = defaultdict(lambda: {'f1': 0, 'accuracy': 0})

    model.eval()
    with torch.no_grad():
        for bucket_name, (min_len, max_len) in buckets.items():
            true_outcomes = []
            pred_outcomes = []

            for i in range(len(dataset)):
                example = dataset[i]
                trace_length = (example['trace'] != 0).sum().item()

                if min_len <= trace_length and (max_len is None or trace_length <= max_len):
                    # Get predictions for this bucket
                    trace = example['trace'].unsqueeze(0).to(device)
                    mask = example['mask'].unsqueeze(0).to(device)
                    times = example['times'].unsqueeze(0).to(device)
                    customer_type = example['customer_type'].unsqueeze(0).to(device)
                    loop_features = example['loop_features'].unsqueeze(0).to(device)

                    # Ensure correct feature dimension
                    expected_feat_dim = model.config.total_feature_dim
                    if loop_features.shape[1] != expected_feat_dim:
                        if loop_features.shape[1] < expected_feat_dim:
                            padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                            loop_features = torch.cat([loop_features, padding], dim=1)
                        else:
                            loop_features = loop_features[:, :expected_feat_dim]

                    _, outcome_logits = model(trace, mask, times, loop_features, customer_type)
                    pred_outcomes.append(torch.argmax(outcome_logits, dim=1).cpu().item())
                    true_outcomes.append(example['outcome'].item())

            if true_outcomes:
                bucket_metrics[bucket_name]['f1'] = f1_score(true_outcomes, pred_outcomes, average='macro',
                                                             zero_division=0)
                bucket_metrics[bucket_name]['accuracy'] = accuracy_score(true_outcomes, pred_outcomes)

    # Create plot
    bucket_names = list(bucket_metrics.keys())
    f1_scores = [bucket_metrics[b]['f1'] for b in bucket_names]
    accuracies = [bucket_metrics[b]['accuracy'] for b in bucket_names]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(bucket_names))
    width = 0.35

    plt.bar(x - width / 2, f1_scores, width, label='F1 Score')
    plt.bar(x + width / 2, accuracies, width, label='Accuracy')

    plt.xlabel('Bucket')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Trace Length Bucket')
    plt.xticks(x, bucket_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('bucket_performance.png')
    plt.close()

# Create all three plots
plot_prefix_performance(model, pytorch_dataset, device)
plot_epoch_performance('training.log')
plot_bucket_performance(model, pytorch_dataset, device)
