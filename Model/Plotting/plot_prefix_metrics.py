import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import seaborn as sns
from sys import argv
import sys
sys.path.append('..')
from dataloader import TraceDataset
from model import MultitaskBERTModel
from train import Config


def evaluate_per_prefix(model, dataset, device, max_prefix_length=None):
    """
    Evaluate model performance for different prefix lengths, averaging across all traces
    """
    model.eval()
    prefix_metrics = defaultdict(lambda: {
        'next_activity': {'true': [], 'pred': [], 'count': 0},
        'outcome': {'true': [], 'pred': [], 'count': 0}
    })
    
    # First, find max trace length in dataset
    max_trace_length = max((example['trace'] != 0).sum().item() for example in dataset)
    if max_prefix_length is None:
        max_prefix_length = max_trace_length
    
    with torch.no_grad():
        for prefix_len in range(1, max_prefix_length + 1):
            # For each prefix length, evaluate all traces that are long enough
            for i in range(len(dataset)):
                example = dataset[i]
                trace_length = (example['trace'] != 0).sum().item()
                
                if trace_length >= prefix_len:
                    # Get predictions using current prefix length
                    trace = example['trace'][:prefix_len].unsqueeze(0).to(device)
                    mask = example['mask'][:prefix_len].unsqueeze(0).to(device)
                    times = example['times'][:prefix_len].unsqueeze(0).to(device)
                    next_activity = example['next_activity'][:prefix_len].to(device)
                    customer_type = example.get('customer_type', torch.zeros(1)).unsqueeze(0).to(device)
                
                # Create zero tensor for loop_features if not present
                if 'loop_features' in example:
                    loop_features = example['loop_features'].unsqueeze(0).to(device)
                else:
                    expected_feat_dim = model.config.total_feature_dim
                    loop_features = torch.zeros(1, expected_feat_dim, device=device)
                
                # Get predictions
                next_activity_logits, outcome_logits = model(
                    trace, mask, times, loop_features, customer_type
                )
                
                # Store next activity predictions
                valid_positions = mask[0].bool()
                true_activities = next_activity[valid_positions].cpu().tolist()
                pred_activities = torch.argmax(next_activity_logits[0], dim=1)[valid_positions].cpu().tolist()

                # Store predictions and count samples
                prefix_metrics[prefix_len]['next_activity']['true'].extend(true_activities)
                prefix_metrics[prefix_len]['next_activity']['pred'].extend(pred_activities)
                prefix_metrics[prefix_len]['outcome']['true'].append(example['outcome'].item())
                prefix_metrics[prefix_len]['outcome']['pred'].append(
                    torch.argmax(outcome_logits, dim=1).cpu().item()
                )
                prefix_metrics[prefix_len]['next_activity']['count'] += len(true_activities)
                prefix_metrics[prefix_len]['outcome']['count'] += 1

    return prefix_metrics
    
    return prefix_metrics

def compute_metrics(prefix_metrics):
    """
    Compute accuracy and F1 score for each prefix length
    """
    results = defaultdict(dict)
    
    for prefix_len, metrics in prefix_metrics.items():
        # Next Activity metrics
        if len(metrics['next_activity']['true']) > 0:
            results[prefix_len]['next_activity_f1'] = f1_score(
                metrics['next_activity']['true'],
                metrics['next_activity']['pred'],
                average='macro',
                zero_division=0
            )
            results[prefix_len]['next_activity_acc'] = accuracy_score(
                metrics['next_activity']['true'],
                metrics['next_activity']['pred']
            )
        
        # Outcome metrics
        if len(metrics['outcome']['true']) > 0:
            results[prefix_len]['outcome_f1'] = f1_score(
                metrics['outcome']['true'],
                metrics['outcome']['pred'],
                average='macro',
                zero_division=0
            )
            results[prefix_len]['outcome_acc'] = accuracy_score(
                metrics['outcome']['true'],
                metrics['outcome']['pred']
            )
    
    return results


def plot_metrics(results, save_path='metrics_plots'):
    """
    Generate and save plots for accuracy and F1 score
    """
    # Import and set seaborn style properly
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('paper')
    sns.set_palette('deep')
    
    # Prepare data
    prefix_lengths = sorted(results.keys())
    metrics = {
        'Accuracy': {'next_activity': [], 'outcome': []},
        'F1 Score': {'next_activity': [], 'outcome': []}
    }
    
    for prefix_len in prefix_lengths:
        metrics['Accuracy']['next_activity'].append(results[prefix_len]['next_activity_acc'])
        metrics['Accuracy']['outcome'].append(results[prefix_len]['outcome_acc'])
        metrics['F1 Score']['next_activity'].append(results[prefix_len]['next_activity_f1'])
        metrics['F1 Score']['outcome'].append(results[prefix_len]['outcome_f1'])
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for idx, metric_name in enumerate(['Accuracy', 'F1 Score']):
        ax = axes[idx]
        
        # Plot lines with seaborn color palette
        ax.plot(prefix_lengths, metrics[metric_name]['next_activity'], 
                marker='o', label='Next Activity Prediction', linewidth=2)
        ax.plot(prefix_lengths, metrics[metric_name]['outcome'], 
                marker='s', label='Outcome Prediction', linewidth=2)
        
        # Customize plot
        ax.set_xlabel('Prefix Length')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Prefix Length')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.close()



def load_model_with_fixes(model, checkpoint_path):
    # Load the state dict
    state_dict = torch.load(checkpoint_path)
    
    # Remove unexpected e-prompt keys
    keys_to_remove = [
        "prompted_bert.e_prompt.task_storage.1",
        "prompted_bert.e_prompt.task_storage.2", 
        "prompted_bert.e_prompt.task_storage.3"
    ]
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    
    # Handle size mismatch for outcome head
    if 'outcome_head.1.weight' in state_dict:
        current_weight = model.outcome_head[1].weight
        saved_weight = state_dict['outcome_head.1.weight']
        
        # Truncate or pad the saved weights to match current model
        if saved_weight.size(1) > current_weight.size(1):
            state_dict['outcome_head.1.weight'] = saved_weight[:, :current_weight.size(1)]
        else:
            padding = torch.zeros(current_weight.size(0), 
                                current_weight.size(1) - saved_weight.size(1),
                                device=saved_weight.device)
            state_dict['outcome_head.1.weight'] = torch.cat([saved_weight, padding], dim=1)
    
    # Load the modified state dict
    model.load_state_dict(state_dict, strict=False)
    return model

def main():
    # Load your model and dataset
    log = argv[1]  # Replace with your log name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and model
    pytorch_dataset = torch.load(f'../datasets/{log}/pytorch_dataset_consistent.pt')
    pretrained_weights = f"../datasets/{log}/mam_pretrained_model"
    
    # Initialize model
    config = Config(pytorch_dataset)  # You'll need to import Config
    model = MultitaskBERTModel(config, pretrained_weights=pretrained_weights).to(device)
    # Load trained model weights with fixes
    model = load_model_with_fixes(model, f'../datasets/{log}/multitask_bert_model.pth')

    # Load trained model weights
    # model.load_state_dict(torch.load(f'../datasets/{log}/multitask_bert_model.pth'))
    
    # Evaluate per prefix length
    prefix_metrics = evaluate_per_prefix(model, pytorch_dataset, device)
    
    # Compute metrics
    results = compute_metrics(prefix_metrics)
    
    # Generate plots
    plot_metrics(results, save_path=f'../datasets/{log}/prefix_metrics')
    
    print("Plots have been generated and saved!")

if __name__ == "__main__":
    main()
