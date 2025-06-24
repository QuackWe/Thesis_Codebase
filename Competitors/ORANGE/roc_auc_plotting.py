import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True, help='Base output directory')
parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
args = parser.parse_args()
dataset_name = args.log
output_dir = args.output_dir

def create_metric_plot(df, metric_name, output_filename):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    # Main plot
    ax.plot(df['LENGHT'], df[metric_name], 
           marker='o', linewidth=2, markersize=8,
           zorder=3)
    
    # Titles and labels
    plt.title(f'{metric_name} vs Sequence Length', fontsize=14, pad=20)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    
    # Grid customization
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Sparse value labels
    label_every = 5  # Show ~5 labels max
    y_offset = 10
    for idx, (x, y) in enumerate(zip(df['LENGHT'], df[metric_name])):
        if idx % label_every == 0:
            ax.annotate(f'{y:.3f}', 
                       (x, y),
                       textcoords="offset points",
                       xytext=(0, y_offset),
                       ha='center',
                       fontsize=9,
                       rotation=45,
                       alpha=0.7)
            y_offset *= -1  # Alternate label positions
    
    # X-axis optimization
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    plt.xticks(df['LENGHT'][::2])  # Show every other tick
    
    # Y-axis limits
    y_min, y_max = df[metric_name].min(), df[metric_name].max()
    plt.ylim(y_min - 0.02, y_max + 0.02)
    
    # Secondary axis for samples
    ax2 = ax.twinx()
    ax2.plot(df['LENGHT'], df['NUMBEROFSAMPLES'], 
            color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Number of Samples', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f'Figure {metric_name} saved!')
    plt.close()

# Style settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Generate plots
df = pd.read_csv(f"{output_dir}/{dataset_name}_results.csv", sep=';')
create_metric_plot(df, 'ROC_AUC_SCORE', f'{output_dir}/roc_auc_scores.png')
create_metric_plot(df, 'F1_SCORE', f'{output_dir}/f1_scores.png')
create_metric_plot(df, 'ACCURACY', f'{output_dir}/accuracy_scores.png')
