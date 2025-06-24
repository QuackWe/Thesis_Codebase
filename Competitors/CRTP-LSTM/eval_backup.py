import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import BagDataGenerator, read_data
import helpers

def evaluate_model(model, test_generator, prefix_lengths):
    results = {
        'prefix_length': [],
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'num_samples': []
    }
    
    # Get predictions for full sequences
    predictions = model.predict(test_generator)
    
    # Get true labels from generator
    true_labels = []
    for i in range(len(test_generator)):
        _, batch_labels = test_generator[i]
        true_labels.append(batch_labels['trace_out'])  # Access trace_out directly
    
    true_labels = np.concatenate(true_labels, axis=0)
    
    # For each prefix length
    for length in prefix_lengths:
        # Truncate predictions and labels
        truncated_preds = predictions[:, :length, :]
        truncated_labels = true_labels[:, :length, :]
        
        # Convert to class predictions
        pred_classes = np.argmax(truncated_preds, axis=-1).reshape(-1)
        true_classes = np.argmax(truncated_labels, axis=-1).reshape(-1)
        
        # Store results
        results['prefix_length'].append(length)
        results['accuracy'].append(accuracy_score(true_classes, pred_classes))
        results['f1'].append(f1_score(true_classes, pred_classes, average='weighted'))
        results['precision'].append(precision_score(true_classes, pred_classes, 
                                                average='weighted', 
                                                zero_division=0))
        results['recall'].append(recall_score(true_classes, pred_classes, 
                                            average='weighted', 
                                            zero_division=0))
        results['num_samples'].append(len(true_classes))
    
    # Calculate overall metrics
    overall_pred_classes = np.argmax(predictions, axis=-1).reshape(-1)
    overall_true_classes = np.argmax(true_labels, axis=-1).reshape(-1)
    
    overall_metrics = {
        'accuracy': accuracy_score(overall_true_classes, overall_pred_classes),
        'f1': f1_score(overall_true_classes, overall_pred_classes, average='weighted'),
        'precision': precision_score(overall_true_classes, overall_pred_classes, 
                                average='weighted', zero_division=0),
        'recall': recall_score(overall_true_classes, overall_pred_classes, 
                            average='weighted', zero_division=0)
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
    from sys import argv
    log = argv[1]
    # Load your model and data
    model = tf.keras.models.load_model(f'datasets/{log}/model_checkpoint.h5')
    data_directory = f"./datasets/{log}/"
    data, _, _, data_test = read_data(data_directory)
    # Feature Dictionary (Removed time features)
    feat_dic = {
        'cat_feat': ['cat_column1', 'cat_column2'],
        'num_feat': ['num_column1', 'num_column2']
    }

    # Generate Helpers
    helpers_dic = helpers.get_helpers(data, feat_dic)
    
    # Initialize test generator (same as in train.py)
    test_generator = BagDataGenerator(data_frame=data_test,
                                    output_dim=50,
                                    feat_dic=feat_dic,
                                    helpers_dic=helpers_dic,
                                    batch_size=128,
                                    shuffle=False)
    
    # Define prefix lengths to evaluate
    prefix_lengths = range(1, 180)
    
    # Evaluate model
    results, overall_metrics = evaluate_model(model, test_generator, prefix_lengths)
    
    # Save results to file
    with open(f'datasets/{log}/evaluation_results.txt', 'w') as f:
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
    plot_metrics(results, 'accuracy', f'datasets/{log}/accuracy_per_prefix.png')
    plot_metrics(results, 'f1', f'datasets/{log}/f1_per_prefix.png')
