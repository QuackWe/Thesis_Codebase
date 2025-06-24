import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import BagDataGenerator, read_data
import helpers
import os
import argparse

def evaluate_model(model, test_generator, prefix_lengths, data_test):
    """Evaluate model and save predictions in standardized format"""
    # Get predictions and true labels
    predictions = model.predict(test_generator)
    true_labels = np.concatenate([batch[1]['trace_out'] for batch in test_generator], axis=0)
    
    # Get actual trace lengths from test data
    trace_lengths = data_test['trace'].apply(lambda x: len(x.split(', '))).values
    
    # Store all predictions in standardized format
    all_predictions = {
        'prefix_length': [],
        'activity_true': [],
        'activity_pred': [],
        'activity_probs': []
    }
    
    # Collect predictions for all traces and positions
    for i, trace_len in enumerate(trace_lengths):
        for pos in range(min(trace_len, predictions.shape[1])):
            all_predictions['prefix_length'].append(pos + 1)
            all_predictions['activity_true'].append(np.argmax(true_labels[i, pos]))
            all_predictions['activity_pred'].append(np.argmax(predictions[i, pos]))
            all_predictions['activity_probs'].append(predictions[i, pos])
    
    # Convert to DataFrame
    results_df = pd.DataFrame({
        'prefix_length': all_predictions['prefix_length'],
        'activity_true': all_predictions['activity_true'],
        'activity_pred': all_predictions['activity_pred']
    })
    
    # Add probability columns for each activity class
    activity_probs = np.array(all_predictions['activity_probs'])
    for i in range(activity_probs.shape[1]):
        results_df[f'activity_prob_{i}'] = activity_probs[:, i]
    
    # Add dummy columns for outcome prediction (since this is activity-only model)
    results_df['outcome_true'] = -1  # Use -1 to indicate N/A
    results_df['outcome_pred'] = -1
    results_df['outcome_prob_0'] = np.nan
    results_df['outcome_prob_1'] = np.nan
    
    # Save predictions to CSV
    results_df.to_csv(f"{output_dir}/predictions.csv", index=False)
    print(f"Predictions saved to {output_dir}/predictions.csv")
    
    # Calculate metrics per prefix length for backward compatibility
    results = {
        'prefix_length': [],
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'num_samples': []
    }

    # Get predictions and true labels
    predictions = model.predict(test_generator)
    true_labels = np.concatenate([batch[1]['trace_out'] for batch in test_generator], axis=0)
    
    # Get actual trace lengths from test data
    trace_lengths = data_test['trace'].apply(lambda x: len(x.split(', '))).values

    for length in prefix_lengths:
        # Find traces long enough for this prefix length
        valid_mask = trace_lengths >= length
        num_samples = np.sum(valid_mask)
        
        if num_samples == 0:
            # Handle case with no valid samples
            results['prefix_length'].append(length)
            results['accuracy'].append(0)
            results['f1'].append(0)
            results['precision'].append(0)
            results['recall'].append(0)
            results['num_samples'].append(0)
            continue

        # Extract predictions and labels for valid traces at this position
        pred_at_position = predictions[valid_mask, length-1, :]
        true_at_position = true_labels[valid_mask, length-1, :]
        
        # Convert to class predictions
        pred_classes = np.argmax(pred_at_position, axis=-1)
        true_classes = np.argmax(true_at_position, axis=-1)

        # Store results
        results['prefix_length'].append(length)
        results['accuracy'].append(accuracy_score(true_classes, pred_classes))
        results['f1'].append(f1_score(true_classes, pred_classes, average='weighted'))
        results['precision'].append(precision_score(true_classes, pred_classes, 
                                                  average='weighted', zero_division=0))
        results['recall'].append(recall_score(true_classes, pred_classes, 
                                            average='weighted', zero_division=0))
        results['num_samples'].append(num_samples)

    # Calculate overall metrics (trace-wise average)
    overall_metrics = {
        'accuracy': np.mean(results['accuracy']),
        'f1': np.mean(results['f1']),
        'precision': np.mean(results['precision']),
        'recall': np.mean(results['recall'])
    }

    return results, overall_metrics



def plot_metrics(results, metric_name, output_file):
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8')
    
    ax = plt.gca()
    line = ax.plot(results['prefix_length'], results[metric_name], 
                 marker='o', linewidth=2, markersize=8)
    
    plt.title(f'{metric_name.capitalize()} vs Prefix Length', fontsize=14, pad=20)
    plt.xlabel('Prefix Length', fontsize=12)
    plt.ylabel(metric_name.capitalize(), fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Improved annotation with spacing
    for i, (x, y) in enumerate(zip(results['prefix_length'], results[metric_name])):
        # Only label every other point if many data points
        if i % 5 == 0:  # Adjust this number based on your data density
            ax.annotate(f'{y:.3f}', 
                        (x, y), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=9,
                        rotation=45,
                        alpha=0.7)
    
    # Add vertical lines for better readability
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add number of samples as secondary axis
    ax2 = ax.twinx()
    ax2.plot(results['prefix_length'], results['num_samples'], 
           color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Number of Samples', color='gray')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Plot {metric_name} saved!')
    plt.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='Base output directory')
    parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
    args = parser.parse_args()
    log = args.log
    output_dir = args.output_dir

    # Load your model and data
    model = tf.keras.models.load_model(f'{output_dir}/model_checkpoint.h5')
    data_directory = output_dir
    data, _, _, data_test = read_data(data_directory+'/')
    # Feature Dictionary (Removed time features)
    feat_dic = {
        'cat_feat': ['cat_column1', 'cat_column2'],
        'num_feat': ['num_column1', 'num_column2']
    }

    # Generate Helpers
    helpers_dic = helpers.get_helpers(data, feat_dic)
    
    # Initialize test generator (same as in train.py)
    test_generator = BagDataGenerator(data_frame=data_test,
                                    output_dim=300,
                                    feat_dic=feat_dic,
                                    helpers_dic=helpers_dic,
                                    batch_size=128,
                                    shuffle=False)
    
    # Define prefix lengths to evaluate
    prefix_lengths = range(1, 300)
    
    # Evaluate model
    results, overall_metrics = evaluate_model(model, test_generator, prefix_lengths, data_test)
    
    # Save results to file
    with open(f'{output_dir}/evaluation_results.txt', 'w') as f:
        f.write("Overall Metrics:\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        f.write("\nMetrics per prefix length:\n")
        f.write("Length;Accuracy;F1;Precision;Recall;NumSamples\n")
        for i in range(len(results['prefix_length'])):
            f.write(f"{results['prefix_length'][i]};"
                f"{results['accuracy'][i]:.4f};"
                f"{results['f1'][i]:.4f};"
                f"{results['precision'][i]:.4f};"
                f"{results['recall'][i]:.4f};"
                f"{results['num_samples'][i]}\n")

    
    # Create plots
    plot_metrics(results, 'accuracy', f'{output_dir}/accuracy_per_prefix.png')
    plot_metrics(results, 'f1', f'{output_dir}/f1_per_prefix.png')
