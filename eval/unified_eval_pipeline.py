import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sys import argv
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    if model_name != "ModernBERT":
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
            
def load_class_distributions(dataset_name, model_name, run_dir=None, config_dir=None):
    """Load class distribution data for plotting with new directory structure."""
    base_path = f"../Model_test/datasets/{dataset_name}"
    
    try:
        if model_name == "ModernBERT":
            if run_dir and config_dir:
                dist_path = os.path.join(base_path, run_dir, config_dir)
                activity_dist = pd.read_csv(os.path.join(dist_path, 'activity_class_distribution.csv'), index_col=0)
                outcome_dist = pd.read_csv(os.path.join(dist_path, 'outcome_class_distribution.csv'), index_col=0)
            else:
                return None, None
        else:
            base_path = f"../Competitors/{model_name}/datasets/{dataset_name}"
            activity_dist = pd.read_csv(f"{base_path}/activity_class_distribution.csv", index_col=0)
            outcome_dist = pd.read_csv(f"{base_path}/outcome_class_distribution.csv", index_col=0)
        return activity_dist, outcome_dist
    except Exception as e:
        print(f"Error loading class distributions: {str(e)}")
        return None, None
    
def get_metric(prefix_metrics, metric):
    """Return lists of prefix lengths, metric values, and sample counts."""
    if not prefix_metrics or not isinstance(prefix_metrics, list):
        return [], [], []
    
    # Convert list of metrics to proper format
    xs = [m['Length'] for m in prefix_metrics]
    ys = [m[metric] for m in prefix_metrics]
    samples = [m['NumSamples'] for m in prefix_metrics]
    
    return xs, ys, samples

def create_metric_plot(ax, x_my, y_my, dist_my, x_comp, y_comp, dist_comp, metric_name, comp_name, my_name="ModernBERT", task="activity"):
    """Create a styled plot with stacked bar sample counts on secondary axis"""
    # Main metrics lines
    if x_comp and y_comp:
        ax.plot(x_comp, y_comp, marker='o', linewidth=2, markersize=8,
                color='green' if comp_name == "CRTP-LSTM" else 'orange' if comp_name == "ORANGE" else 'red',
                label=comp_name, zorder=3)
    ax.plot(x_my, y_my, marker='s', linewidth=2, markersize=8,
            color='blue', label=my_name, zorder=3)
    
    # Titles and labels
    ax.set_title(f'{metric_name} vs Prefix Length', fontsize=14, pad=20)
    ax.set_xlabel('Prefix Length', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Value labels for metrics
    label_every = max(1, len(x_comp) // 5)
    y_offset = 10
    for idx, (x, y) in enumerate(zip(x_comp, y_comp)):
        if idx % label_every == 0:
            ax.annotate(f'{y:.3f}', (x, y),
                      textcoords="offset points",
                      xytext=(0, y_offset),
                      ha='center', fontsize=9, alpha=0.7)
            y_offset *= -1

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    
    # Y-axis limits for metrics
    all_y = (y_comp + y_my) if (x_comp and x_my) else (y_comp if x_comp else y_my)
    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        ax.set_ylim(max(0, y_min - 0.05), min(1.0, y_max + 0.05))
    
    # Secondary axis for sample distribution
    ax2 = ax.twinx()
    
    # Set up bars
    bar_width = 0.35
    class_columns = [col for col in dist_my.columns if f'{task}_class_' in col]
    n_classes = len(class_columns)
    
    # Create color palette for classes
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    # Plot stacked bars for ModernBERT
    bottom_my = np.zeros(len(x_my))
    for i, col in enumerate(class_columns):
        values = dist_my[col].values
        ax2.bar(np.array(x_my) - bar_width/2, values, bar_width,
                bottom=bottom_my, color=colors[i], alpha=0.5,
                label=f'{my_name} {col.split("_")[-1]}')
        bottom_my += values
    
    # Plot stacked bars for competitor
    if dist_comp is not None:
        bottom_comp = np.zeros(len(x_comp))
        for i, col in enumerate(class_columns):
            values = dist_comp[col].values
            ax2.bar(np.array(x_comp) + bar_width/2, values, bar_width,
                   bottom=bottom_comp, color=colors[i], alpha=0.5,
                   label=f'{comp_name} {col.split("_")[-1]}')
            bottom_comp += values
    
    ax2.set_ylabel('Number of Samples', color='gray')
    
    # Legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, 
             loc='upper center', bbox_to_anchor=(0.5, -0.15), 
             ncol=3, fontsize=10)

def create_comparison_plots(dataset_name, modernbert_metrics, competitor_metrics):
    """Create comparison plots for each ModernBERT configuration."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # For each ModernBERT configuration
    for config_key, mb_metrics in modernbert_metrics.items():
        run_id, config = config_key.split('/')
        plot_dir = f"./results/{dataset_name}/{run_id}/{config}/plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Load class distributions for this configuration
        modernbert_act_dist, modernbert_out_dist = load_class_distributions(
            dataset_name, "ModernBERT", run_id, config
        )
        
        # Create comparison plots for each competitor
        for competitor, comp_metrics in competitor_metrics.items():
            if not comp_metrics.get('default'):
                continue
        
        # Load class distributions
        modernbert_act_dist, modernbert_out_dist = load_class_distributions(dataset_name, "ModernBERT")
        competitor_act_dist, competitor_out_dist = load_class_distributions(dataset_name, competitor)
            
        # Activity prediction plots
        if 'activity' in metrics and metrics['activity'] and 'activity' in modernbert_metrics and modernbert_metrics['activity']:
            try:
                fig1, axs1 = plt.subplots(1, 2, figsize=(16, 7))
            
                my_metrics = modernbert_metrics['activity'][1]
                comp_metrics = metrics['activity'][1]
                
                # Extract metrics
                x_acc_my, y_acc_my, _ = get_metric(my_metrics, "Accuracy")
                x_acc_comp, y_acc_comp, _ = get_metric(comp_metrics, "Accuracy")
                x_f1_my, y_f1_my, _ = get_metric(my_metrics, "F1")
                x_f1_comp, y_f1_comp, _ = get_metric(comp_metrics, "F1")
                
                if x_acc_my and x_acc_comp:
                    create_metric_plot(axs1[0], x_acc_my, y_acc_my, modernbert_act_dist, 
                                    x_acc_comp, y_acc_comp, competitor_act_dist,
                                    "Accuracy", competitor, task="activity")
                    create_metric_plot(axs1[1], x_f1_my, y_f1_my, modernbert_act_dist,
                                    x_f1_comp, y_f1_comp, competitor_act_dist,
                                    "F1 Score", competitor, task="activity")
                    
                    plt.tight_layout()
                    plt.savefig(f'{plot_dir}/nap_{competitor.lower()}_comparison.png', 
                              dpi=300, bbox_inches='tight')
                    print(f"Saved activity prediction plots for {competitor}")
                plt.close(fig1)
            except Exception as e:
                print(f"Error creating activity plots for {competitor}: {str(e)}")
        
        # Outcome prediction plots
        if 'outcome' in metrics and metrics['outcome'] and 'outcome' in modernbert_metrics and modernbert_metrics['outcome']:
            try:
                fig2, axs2 = plt.subplots(1, 3, figsize=(21, 7))
                
                my_metrics = modernbert_metrics['outcome'][1]
                comp_metrics = metrics['outcome'][1]
                
                # Extract metrics
                x_acc_my, y_acc_my, _ = get_metric(my_metrics, "Accuracy")
                x_acc_comp, y_acc_comp, _ = get_metric(comp_metrics, "Accuracy")
                x_f1_my, y_f1_my, _ = get_metric(my_metrics, "F1")
                x_f1_comp, y_f1_comp, _ = get_metric(comp_metrics, "F1")
                x_roc_my, y_roc_my, _ = get_metric(my_metrics, "ROC_AUC")
                x_roc_comp, y_roc_comp, _ = get_metric(comp_metrics, "ROC_AUC")
                
                if x_acc_my and x_acc_comp:
                    create_metric_plot(axs2[0], x_acc_my, y_acc_my, modernbert_out_dist,
                                    x_acc_comp, y_acc_comp, competitor_out_dist,
                                    "Accuracy", competitor, task="outcome")
                    create_metric_plot(axs2[1], x_f1_my, y_f1_my, modernbert_out_dist,
                                    x_f1_comp, y_f1_comp, competitor_out_dist,
                                    "F1 Score", competitor, task="outcome")
                    create_metric_plot(axs2[2], x_roc_my, y_roc_my, modernbert_out_dist,
                                    x_roc_comp, y_roc_comp, competitor_out_dist,
                                    "ROC AUC", competitor, task="outcome")
                    
                    plt.tight_layout()
                    plt.savefig(f'{plot_dir}/outcome_{competitor.lower()}_comparison.png',
                              dpi=300, bbox_inches='tight')
                    print(f"Saved outcome prediction plots for {competitor}")
                plt.close(fig2)
            except Exception as e:
                print(f"Error creating outcome plots for {competitor}: {str(e)}")

def main():    
    dataset_name = argv[1]
        
    # Evaluate ModernBERT with all configurations
    modernbert_results = evaluate_predictions(dataset_name, "ModernBERT")
    
    # Evaluate competitors
    competitor_results = {
        "CRTP-LSTM": evaluate_predictions(dataset_name, "CRTP-LSTM"),
        "ORANGE": evaluate_predictions(dataset_name, "ORANGE"),
        "BERT": evaluate_predictions(dataset_name, "BERT")
    }
    
    # Create comparison plots for each ModernBERT configuration
    create_comparison_plots(dataset_name, modernbert_results, competitor_results)

if __name__ == "__main__":
    main()