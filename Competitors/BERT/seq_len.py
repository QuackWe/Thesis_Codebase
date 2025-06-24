import pandas as pd

def get_max_sequence_lengths(log_name):
    # 1. Original log before any processing
    event_log_file = f'datasets/{log_name}/{log_name}_processed.csv'
    df = pd.read_csv(event_log_file, parse_dates=['Timestamp'])
    df = df.sort_values(by=['CaseID', 'Timestamp'])
    
    # Group by CaseID to get list of activities per trace
    traces = df.groupby('CaseID')['Activity'].apply(list)
    max_seq_len = traces.apply(len).max()
    print(f"Max sequence length in the entire log before tokenization: {max_seq_len}")
    
    # 2. Calculate max prefix length in train/val/test splits
    train_df = pd.read_csv(f'datasets/{log_name}/outcome_train.csv')
    val_df = pd.read_csv(f'datasets/{log_name}/outcome_val.csv')
    test_df = pd.read_csv(f'datasets/{log_name}/outcome_test.csv')
    
    # Convert string to list and get max length
    train_max_len = train_df['Prefix'].apply(eval).apply(len).max()
    val_max_len = val_df['Prefix'].apply(eval).apply(len).max()
    test_max_len = test_df['Prefix'].apply(eval).apply(len).max()
    
    print(f"Max prefix length in train split: {train_max_len}")
    print(f"Max prefix length in validation split: {val_max_len}")
    print(f"Max prefix length in test split: {test_max_len}")
    
    # 3. Additional statistics
    train_mean_len = train_df['Prefix'].apply(eval).apply(len).mean()
    val_mean_len = val_df['Prefix'].apply(eval).apply(len).mean()
    test_mean_len = test_df['Prefix'].apply(eval).apply(len).mean()
    
    print(f"\nAverage prefix length in train split: {train_mean_len:.2f}")
    print(f"Average prefix length in validation split: {val_mean_len:.2f}")
    print(f"Average prefix length in test split: {test_mean_len:.2f}")
    
    # 4. Distribution of prefix lengths
    print("\nPrefix length distribution in train split:")
    length_counts = train_df['Prefix'].apply(eval).apply(len).value_counts().sort_index()
    for length, count in length_counts.items():
        print(f"  Length {length}: {count} prefixes")

# Example usage
log_name = "mortgages"  # Replace with your actual log name
get_max_sequence_lengths(log_name)
