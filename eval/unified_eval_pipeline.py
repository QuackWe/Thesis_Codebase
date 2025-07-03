import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sys import argv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re

def calculate_metrics(y_true, y_pred, prob_scores=None):
    """Calculate all metrics for a given set of predictions."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Calculate ROC AUC if probability scores are provided
    if prob_scores is not None:
        try:
            n_classes = prob_scores.shape[1] if len(prob_scores.shape) > 1 else 2
            if n_classes == 2:
                # For binary classification, use probability of positive class
                binary_scores = prob_scores[:, 1] if len(prob_scores.shape) > 1 else prob_scores
                metrics['ROC_AUC'] = roc_auc_score(y_true, binary_scores)
            else:
                # For multi-class, use OVR approach
                metrics['ROC_AUC'] = roc_auc_score(y_true, prob_scores, multi_class='ovr')
        except Exception as e:
            # print(f"Error calculating ROC AUC: {str(e)}")
            metrics['ROC_AUC'] = float('nan')
    else:
        metrics['ROC_AUC'] = float('nan')
        
    return metrics

def save_metrics(results_df, true_col, pred_col, prob_scores, output_file, model_name):
    """Save metrics both overall and per prefix length to a file."""
    
    # Calculate overall metrics
    y_true = results_df[true_col].values
    y_pred = results_df[pred_col].values
    overall_metrics = calculate_metrics(y_true, y_pred, prob_scores)
    
    # Calculate metrics per prefix length
    prefix_metrics = []
    for length in sorted(results_df['prefix_length'].unique()):
        subset = results_df[results_df['prefix_length'] == length]
        sub_true = subset[true_col].values
        sub_pred = subset[pred_col].values
        
        # Get probability scores for this subset if available
        if prob_scores is not None:
            sub_indices = results_df['prefix_length'] == length
            sub_probs = np.array(prob_scores)[sub_indices]
        else:
            sub_probs = None
            
        metrics = calculate_metrics(sub_true, sub_pred, sub_probs)
        metrics['Length'] = length
        metrics['NumSamples'] = len(sub_true)
        prefix_metrics.append(metrics)
    
    # Save metrics to file
    with open(output_file, 'w') as f:
        f.write(f"Model: {model_name}\n\n")
        f.write("Overall Metrics:\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nMetrics per prefix length:\n")
        f.write("Length;Accuracy;F1;Precision;Recall;ROC_AUC;NumSamples\n")
        for m in prefix_metrics:
            f.write(f"{int(m['Length'])};{m['Accuracy']:.4f};{m['F1']:.4f};" +
                    f"{m['Precision']:.4f};{m['Recall']:.4f};" +
                    f"{m['ROC_AUC'] if not np.isnan(m['ROC_AUC']) else 'nan'};" +
                    f"{m['NumSamples']}\n")
    
    print(f"\nMetrics saved to {output_file}")
    return overall_metrics, prefix_metrics


def load_all_prediction_files(base_path, model_name):
    """Load all prediction files from all configuration folders."""
    predictions_list = []
    
    # Handle BERT and other competitors differently
    if model_name == "BERT":
        # Look for run directories
        run_dirs = [d for d in os.listdir(base_path) if d.startswith('results_')]
        for run_dir in run_dirs:
            try:
                run_path = os.path.join(base_path, run_dir)
                nap_file = os.path.join(run_path, 'predictions_nap.csv')
                outcome_file = os.path.join(run_path, 'predictions_outcome.csv')
                entry = {'run_id': run_dir}
                if os.path.exists(nap_file):
                    entry['activity'] = pd.read_csv(nap_file)
                if os.path.exists(outcome_file):
                    entry['outcome'] = pd.read_csv(outcome_file)
                if 'activity' in entry or 'outcome' in entry:
                    predictions_list.append(entry)
            except Exception as e:
                print(f"Error loading predictions from {run_dir}: {str(e)}")
    elif model_name != "ModernBERT":
        # Look for run directories
        run_dirs = [d for d in os.listdir(base_path) if d.startswith('results_')]
        for run_dir in run_dirs:
            try:
                run_path = os.path.join(base_path, run_dir)
                pred_file = os.path.join(run_path, 'predictions.csv')
                if os.path.exists(pred_file):
                    df = pd.read_csv(pred_file)
                    df['run_id'] = run_dir
                    if model_name == "BERT":
                        if 'nap' in pred_file:
                            df['outcome_true'] = -1
                        elif 'outcome' in pred_file:
                            df['activity_true'] = -1
                    predictions_list.append(df)
            except Exception as e:
                print(f"Error loading {pred_file}: {str(e)}")
    else:
        # Look for run directories
        run_dirs = [d for d in os.listdir(base_path) if d.startswith('results_')]
        print(f"Found run directories: {run_dirs}")
        
        for run_dir in run_dirs:
            run_path = os.path.join(base_path, run_dir)
            # Look for configuration directories within run directory
            config_dirs = [d for d in os.listdir(run_path) if d.startswith('act_') or d.startswith('out_') or d.startswith('dual_')]
            
            for config_dir in config_dirs:
                try:
                    config_path = os.path.join(run_path, config_dir)
                    pred_file = os.path.join(config_path, f'predictions_{config_dir}.csv')
                    if os.path.exists(pred_file):
                        df = pd.read_csv(pred_file)
                        df['run_id'] = run_dir
                        df['config'] = config_dir
                        predictions_list.append(df)
                except Exception as e:
                    print(f"Error loading {pred_file}: {str(e)}")
    
    return predictions_list


def evaluate_predictions(dataset_name, model_name):
    """Evaluate predictions with support for multiple configurations."""
    print("-" * 50)
    print(f"Evaluating {model_name} on {dataset_name}:")

    # Define paths
    base_path = f"../ModernBERT/datasets/{dataset_name}" if model_name == 'ModernBERT' else f"../Competitors/{model_name}/datasets/{dataset_name}"
    
    # Load all prediction files
    predictions_list = load_all_prediction_files(base_path, model_name)
    if not predictions_list:
        print(f"No valid predictions found for {model_name} on {dataset_name}")
        return {'activity': None, 'outcome': None}

    results = {}
    all_metrics = {'activity': [], 'outcome': []}
    config_metrics = {}  # Store metrics by configuration for ModernBERT
    
    # For ModernBERT, evaluate each configuration separately
    if model_name == "ModernBERT":
        for predictions in predictions_list:
            config = predictions['config'].iloc[0]
            run_id = predictions['run_id'].iloc[0]
            run_number = run_id.split('_')[-1]
            
            # Create configuration-specific output directory
            output_dir = f"./results/{dataset_name}/ModernBERT/{config}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Evaluate this configuration
            run_results = evaluate_single_prediction(
                predictions.drop(['config', 'run_id'], axis=1),
                None,
                output_dir,
                model_name,
                metrics_suffix=f"_{run_number}"
            )
            
            # Store results by configuration
            if config not in config_metrics:
                config_metrics[config] = {'activity': [], 'outcome': []}
            
            if run_results.get('activity'):
                config_metrics[config]['activity'].append(run_results['activity'])
            if run_results.get('outcome'):
                config_metrics[config]['outcome'].append(run_results['outcome'])
            
            results[f"{config}/{run_id}"] = run_results
        
        # Calculate averaged metrics for each configuration
        for config, metrics in config_metrics.items():
            output_dir = f"./results/{dataset_name}/ModernBERT/{config}"
            for task in ['activity', 'outcome']:
                if metrics[task]:
                    calculate_average_metrics(metrics[task], output_dir, task, f"{model_name}_{config}")
    elif model_name == "BERT":
        output_dir = f"./results/{dataset_name}/BERT"
        os.makedirs(output_dir, exist_ok=True)
        for entry in predictions_list:
            run_id = entry['run_id']
            run_suffix = f"_{run_id.split('_')[-1]}"
            run_results = {}
            # Evaluate activity (NAP)
            if 'activity' in entry:
                df = entry['activity']
                run_results['activity'] = evaluate_single_prediction(
                    df,
                    task_type='activity',
                    output_dir=output_dir,
                    model_name=model_name,
                    metrics_suffix=run_suffix
                )['activity']
                all_metrics['activity'].append(run_results['activity'])
            # Evaluate outcome
            if 'outcome' in entry:
                df = entry['outcome']
                run_results['outcome'] = evaluate_single_prediction(
                    df,
                    task_type='outcome',
                    output_dir=output_dir,
                    model_name=model_name,
                    metrics_suffix=run_suffix
                )['outcome']
                all_metrics['outcome'].append(run_results['outcome'])
            results[run_id] = run_results
        # Calculate and save averaged metrics
        for task in ['activity', 'outcome']:
            if all_metrics[task]:
                calculate_average_metrics(all_metrics[task], output_dir, task, model_name)
    else:
        # For competitors, evaluate each run separately
        for predictions in predictions_list:
            run_id = predictions['run_id'].iloc[0]
            output_dir = f"./results/{dataset_name}/{model_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Evaluate this run
            run_results = evaluate_single_prediction(
                predictions.drop(['run_id'], axis=1),
                None,
                output_dir,
                model_name,
                metrics_suffix=f"_{run_id.split('_')[-1]}"  # Add run ID to metrics filename
            )
            
            results[run_id] = run_results
            
            # Store both overall and prefix metrics
            if run_results.get('activity'):
                all_metrics['activity'].append(run_results['activity'])
            if run_results.get('outcome'):
                all_metrics['outcome'].append(run_results['outcome'])
    
        # Calculate and save averaged metrics
        for task in ['activity', 'outcome']:
            if all_metrics[task]:
                calculate_average_metrics(all_metrics[task], output_dir, task, model_name)
    
    return results


def calculate_average_metrics(metrics_list, output_dir, task, model_name):
    """Calculate average and std of metrics across multiple runs for both overall and per-prefix metrics."""
    
    # Extract overall metrics and per-prefix metrics
    overall_metrics = []
    prefix_metrics = {}
    
    for run_metrics in metrics_list:
        # Each run_metrics is a tuple (overall_metrics, prefix_metrics_list)
        overall_metrics.append(run_metrics[0])  # First element is overall metrics dict
        
        # Process prefix metrics
        for prefix_data in run_metrics[1]:  # Second element is list of prefix metrics
            length = prefix_data['Length']
            if length not in prefix_metrics:
                prefix_metrics[length] = []
            prefix_metrics[length].append(prefix_data)
    
    # Calculate statistics for overall metrics
    overall_df = pd.DataFrame(overall_metrics)
    overall_mean = overall_df.mean()
    overall_std = overall_df.std()
    
    # Calculate statistics for per-prefix metrics
    prefix_stats = []
    for length, metrics in sorted(prefix_metrics.items()):
        prefix_df = pd.DataFrame(metrics)
        # Calculate mean and std for all metrics except Length
        metrics_to_analyze = ['Accuracy', 'F1', 'Precision', 'Recall', 'ROC_AUC']
        prefix_means = prefix_df[metrics_to_analyze].mean()
        prefix_stds = prefix_df[metrics_to_analyze].std()
        avg_samples = int(prefix_df['NumSamples'].mean())
        
        prefix_stats.append({
            'Length': length,
            'means': prefix_means,
            'stds': prefix_stds,
            'NumSamples': avg_samples
        })
    
    # Save averaged metrics to file
    output_file = f"{output_dir}/{task}_metrics_averaged.txt"
    with open(output_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Averaged across {len(metrics_list)} runs\n\n")
        
        # Write overall metrics
        f.write("Overall Metrics:\n")
        for metric in overall_mean.index:
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {overall_mean[metric]:.4f}\n")
            f.write(f"  Std:  {overall_std[metric]:.4f}\n")
        
        # Write per-prefix metrics
        f.write("\nMetrics per prefix length:\n")
        f.write("Length;Metric;Mean;Std;AvgSamples\n")
        for stat in prefix_stats:
            length = stat['Length']
            for metric in ['Accuracy', 'F1', 'Precision', 'Recall', 'ROC_AUC']:
                mean_val = stat['means'][metric]
                std_val = stat['stds'][metric]
                if not np.isnan(mean_val):
                    f.write(f"{int(length)};{metric};"
                           f"{mean_val:.4f};"
                           f"{std_val:.4f};"
                           f"{stat['NumSamples']}\n")
    
    print(f"\nAveraged metrics saved to {output_file}")
    return overall_mean, overall_std, prefix_stats
    

def evaluate_single_prediction(predictions, task_type=None, output_dir=None, model_name=None, results=None, metrics_suffix=""):
    """Evaluate a single prediction dataframe."""
    if results is None:
        results = {'activity': None, 'outcome': None}
        
    # Check if probability columns exist
    prob_cols = [col for col in predictions.columns if col.startswith('activity_prob_') 
                or col.startswith('outcome_prob_')]
    has_probs = len(prob_cols) > 0

    # Check if this is a single-task model or use provided task_type
    if task_type == 'activity':
        is_activity_only = True
        is_outcome_only = False
    elif task_type == 'outcome':
        is_activity_only = False
        is_outcome_only = True
    else:
        # Convert to Python bool to avoid numpy.bool_ issues
        is_activity_only = bool(all(predictions['outcome_true'].astype(int) == -1))
        is_outcome_only = bool(all(predictions['activity_true'].astype(int) == -1))
    
    # Evaluate activity predictions if not outcome-only model
    if not is_outcome_only and 'activity_true' in predictions.columns:
        try:
            activity_metrics = save_metrics(
                predictions,
                'activity_true',
                'activity_pred',
                None if not has_probs else predictions[[col for col in prob_cols if col.startswith('activity_prob_')]].values,
                f"{output_dir}/activity_metrics{metrics_suffix}.txt",
                model_name
            )
            results['activity'] = activity_metrics
            print("Activity prediction overall metrics:")
            for metric, value in activity_metrics[0].items():
                print(f"{metric}: {value:.4f}")
        except Exception as e:
            print(f"Error evaluating activity metrics: {str(e)}")
    
    # Evaluate outcome predictions if not activity-only model
    if not is_activity_only and 'outcome_true' in predictions.columns:
        try:
            outcome_metrics = save_metrics(
                predictions,
                'outcome_true',
                'outcome_pred',
                None if not has_probs else predictions[[col for col in prob_cols if col.startswith('outcome_prob_')]].values,
                f"{output_dir}/outcome_metrics{metrics_suffix}.txt",
                model_name
            )
            results['outcome'] = outcome_metrics
            print("Outcome prediction overall metrics:")
            for metric, value in outcome_metrics[0].items():
                print(f"{metric}: {value:.4f}")
        except Exception as e:
            print(f"Error evaluating outcome metrics: {str(e)}")

    print("\nEvaluation complete!")    
    return results
            
def load_class_distributions(dataset_name):
    """
    Load class distribution data for plotting.
    Uses the first available run/config for the dataset, as distributions are identical within the dataset.
    """
    print(f"[DEBUG] Loading class distributions for dataset: {dataset_name}")
    base_dir = f"../ModernBERT/datasets/{dataset_name}"
    try:
        # Find any run/config directory
        for run_dir in os.listdir(base_dir):
            run_path = os.path.join(base_dir, run_dir)
            if not os.path.isdir(run_path):
                continue
            for config_dir in os.listdir(run_path):
                config_path = os.path.join(run_path, config_dir)
                if not os.path.isdir(config_path):
                    continue
                act_file = os.path.join(config_path, f'activity_class_distribution_{config_dir}.csv')
                out_file = os.path.join(config_path, f'outcome_class_distribution_{config_dir}.csv')
                print(f"[DEBUG] Checking for files: {act_file}, {out_file}")
                if os.path.exists(act_file) and os.path.exists(out_file):
                    print(f"[DEBUG] Found class distribution files: {act_file}, {out_file}")
                    activity_dist = pd.read_csv(act_file, index_col=0)
                    outcome_dist = pd.read_csv(out_file, index_col=0)
                    return activity_dist, outcome_dist
        print("[DEBUG] No class distribution files found.")
        return None, None
    except Exception as e:
        print(f"[DEBUG] Error loading class distributions: {str(e)}")
        return None, None
    
def parse_averaged_metrics_file(filepath):
    """
    Parse an averaged metrics file and return a dictionary:
    {
        metric_name: {
            'lengths': [...],
            'means': [...],
            'stds': [...],
            'samples': [...]
        },
        ...
    }
    """
    metrics = {}
    current_metric = None
    with open(filepath, 'r') as f:
        lines = f.readlines()
    in_prefix_section = False
    for line in lines:
        if line.strip().startswith("Metrics per prefix length:"):
            in_prefix_section = True
            continue
        if in_prefix_section:
            if re.match(r'^\d+;', line):
                parts = line.strip().split(';')
                length = int(parts[0])
                metric = parts[1]
                mean = float(parts[2])
                std = float(parts[3])
                samples = int(parts[4])
                if metric not in metrics:
                    metrics[metric] = {'lengths': [], 'means': [], 'stds': [], 'samples': []}
                metrics[metric]['lengths'].append(length)
                metrics[metric]['means'].append(mean)
                metrics[metric]['stds'].append(std)
                metrics[metric]['samples'].append(samples)
    return metrics

def plot_metric_with_ci_and_class_dist(ax, metric_data, label, color, class_dist, task):
    """
    Plot mean and confidence interval for a metric, and add class distribution bars.
    """
    x = metric_data['lengths']
    y = metric_data['means']
    yerr = metric_data['stds']
    ax.plot(x, y, label=label, color=color)
    ax.fill_between(x, [m-s for m, s in zip(y, yerr)], [m+s for m, s in zip(y, yerr)],
                    color=color, alpha=0.2)
    
    # Plot class distribution as stacked bars on secondary axis
    if class_dist is not None:
        ax2 = ax.twinx()
        class_columns = [col for col in class_dist.columns if f'{task}_class_' in col]
        n_classes = len(class_columns)
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        bottom = np.zeros(len(x))
        for i, col in enumerate(class_columns):
            # Align class distribution to prefix lengths
            values = []
            for length in x:
                if str(length) in class_dist.index or int(length) in class_dist.index:
                    values.append(class_dist.loc[length, col])
                else:
                    values.append(0)
            ax2.bar(np.array(x), values, width=0.35, bottom=bottom, color=colors[i], alpha=0.3, label=f'Class {col.split("_")[-1]}')
            bottom += np.array(values)
        ax2.set_ylabel('Num Samples (class dist)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        # Only show legend for the first metric subplot
        if ax.get_subplotspec().colspan.start == 0:
            handles2, labels2 = ax2.get_legend_handles_labels()
            # ax2.legend(handles2, labels2, loc='upper right', fontsize=8)
    else:
        ax2 = None
    return ax, ax2

def plot_comparison_from_averaged(
    dataset_name, 
    modernbert_dir, 
    competitor_dir, 
    task, 
    competitor_name, 
    metrics=['Accuracy', 'F1', 'ROC_AUC'],
    save_path=None,
    class_dist=None
):
    """
    Plot comparison between ModernBERT and a competitor using averaged metrics,
    and add class distribution bars.
    """
    mb_file = f"{modernbert_dir}/{task}_metrics_averaged.txt"
    comp_file = f"{competitor_dir}/{task}_metrics_averaged.txt"
    mb_metrics = parse_averaged_metrics_file(mb_file)
    comp_metrics = parse_averaged_metrics_file(comp_file)
    colors = {'ModernBERT': 'blue', competitor_name: 'orange'}
    plt.figure(figsize=(18, 5))
    for i, metric in enumerate(metrics):
        ax = plt.subplot(1, len(metrics), i+1)
        if metric in mb_metrics:
            plot_metric_with_ci_and_class_dist(ax, mb_metrics[metric], 'ModernBERT', colors['ModernBERT'], class_dist, task)
        if metric in comp_metrics:
            plot_metric_with_ci_and_class_dist(ax, comp_metrics[metric], competitor_name, colors[competitor_name], None, task)
        ax.set_title(f"{metric} vs Prefix Length")
        ax.set_xlabel("Prefix Length")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    if save_path is None:
        save_path = f"./results/{dataset_name}/comparison_{task}_{competitor_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved comparison plot: {save_path}")
    plt.close()


def plot_multi_comparison_from_averaged(
    dataset_name,
    model_dirs,  # dict: {model_name: dir}
    task,
    metrics=['Accuracy', 'F1', 'ROC_AUC'],
    save_path=None,
    class_dist=None,
    colors=None
):
    """
    Plot comparison between multiple models using averaged metrics,
    and add class distribution bars.
    """
    if colors is None:
        colors = {'ModernBERT': 'green', 'BERT': 'blue', 'ORANGE': 'orange', 'CRTP-LSTM': 'red'}
    plt.figure(figsize=(18, 5))
    # Parse metrics for all models
    all_metrics = {}
    for model_name, model_dir in model_dirs.items():
        avg_file = f"{model_dir}/{task}_metrics_averaged.txt"
        if os.path.exists(avg_file):
            all_metrics[model_name] = parse_averaged_metrics_file(avg_file)
        else:
            print(f"[INFO] Skipping {avg_file} (not found)")
    for i, metric in enumerate(metrics):
        ax = plt.subplot(1, len(metrics), i+1)
        for model_name, metrics_dict in all_metrics.items():
            if metric in metrics_dict:
                plot_metric_with_ci_and_class_dist(
                    ax, metrics_dict[metric], model_name, colors.get(model_name, None),
                    class_dist if model_name == "ModernBERT" else None, task
                )
        ax.set_title(f"{metric} vs Prefix Length")
        ax.set_xlabel("Prefix Length")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    if save_path is None:
        save_path = f"./results/{dataset_name}/comparison_{task}_all.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved comparison plot: {save_path}")
    plt.close()
    

def create_comparison_plots_from_averaged(dataset_name):
    """
    For each ModernBERT configuration, create a single comparison plot with all relevant competitors
    using the averaged metrics files for both activity and outcome tasks.
    Also adds class distribution bars from any ModernBERT run/config.
    """
    # Load class distributions once per dataset
    activity_class_dist, outcome_class_dist = load_class_distributions(dataset_name)
    modernbert_base = f"./results/{dataset_name}/ModernBERT"
    for config in os.listdir(modernbert_base):
        config_dir = os.path.join(modernbert_base, config)
        if not os.path.isdir(config_dir):
            continue
        for task in ["activity", "outcome"]:
            mb_avg_file = os.path.join(config_dir, f"{task}_metrics_averaged.txt")
            if not os.path.exists(mb_avg_file):
                print(f"[INFO] Skipping {mb_avg_file} (not found)")
                continue
            # Choose competitors for each task
            if task == "activity":
                competitors = ["BERT", "CRTP-LSTM"]
                metrics_to_plot = ['Accuracy', 'F1']
                class_dist = activity_class_dist
            else:
                competitors = ["BERT", "ORANGE"]
                metrics_to_plot = ['Accuracy', 'F1', 'ROC_AUC']
                class_dist = outcome_class_dist
            # Build model_dirs dict
            model_dirs = {"ModernBERT": config_dir}
            for comp in competitors:
                comp_dir = f"./results/{dataset_name}/{comp}"
                comp_avg_file = os.path.join(comp_dir, f"{task}_metrics_averaged.txt")
                if os.path.exists(comp_avg_file):
                    model_dirs[comp] = comp_dir
                else:
                    print(f"[INFO] Skipping {comp_avg_file} (not found)")
            if len(model_dirs) < 2:
                print(f"[INFO] Not enough models for {task} in {config}, skipping plot.")
                continue
            save_dir = os.path.join(config_dir, "plots")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{task}_comparison_all.png")
            plot_multi_comparison_from_averaged(
                dataset_name=dataset_name,
                model_dirs=model_dirs,
                task=task,
                metrics=metrics_to_plot,
                save_path=save_path,
                class_dist=class_dist
            )
                
def main():    
    dataset_name = argv[1]
        
    # # Evaluate ModernBERT with all configurations
    # modernbert_results = evaluate_predictions(dataset_name, "ModernBERT")
    
    # # Evaluate competitors
    # competitor_results = {
    #     "CRTP-LSTM": evaluate_predictions(dataset_name, "CRTP-LSTM"),
    #     "ORANGE": evaluate_predictions(dataset_name, "ORANGE"),
    #     "BERT": evaluate_predictions(dataset_name, "BERT")
    # }
    
    # Create comparison plots for each ModernBERT configuration
    create_comparison_plots_from_averaged(dataset_name)
if __name__ == "__main__":
    main()