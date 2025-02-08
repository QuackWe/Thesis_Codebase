import matplotlib.pyplot as plt
import numpy as np
import torch
from sys import argv
import sys
sys.path.append('..')  # Add parent directory to path
from dataloader import TraceDataset  # Import the dataset class

def analyze_trace_lengths(dataset):
    """Create histogram of trace lengths from dataset"""
    # Calculate actual trace lengths (excluding padding)
    trace_lengths = [(trace != 0).sum().item() for trace in dataset.traces]

    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(trace_lengths, bins='auto', edgecolor='black')
    plt.title('Distribution of Trace Lengths')
    plt.xlabel('Trace Length')
    plt.ylabel('Frequency')

    # Add statistics
    mean_length = np.mean(trace_lengths)
    median_length = np.median(trace_lengths)
    print(f"\n=== Trace Length Statistics ===")
    print(f"Mean length: {mean_length:.2f}")
    print(f"Median length: {median_length:.2f}")
    print(f"Min length: {min(trace_lengths)}")
    print(f"Max length: {max(trace_lengths)}")

    # Add vertical lines for mean and median
    plt.axvline(mean_length, color='r', linestyle='--', label=f'Mean ({mean_length:.1f})')
    plt.axvline(median_length, color='g', linestyle='--', label=f'Median ({median_length:.1f})')
    plt.legend()

    plt.grid(True, alpha=0.3)
    plt.show()


log = argv[1]
pytorch_dataset = torch.load(f'../datasets/{log}/pytorch_dataset_consistent.pt')

# Call the function with your dataset
analyze_trace_lengths(pytorch_dataset)
