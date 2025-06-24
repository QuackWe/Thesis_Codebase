import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
import re
import argparse
import os
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True, help='Base output directory')
parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
args = parser.parse_args()
log = args.log
output_dir = args.output_dir

trainval_file = f"../../datasets/{log}/{log}_train-val.csv"
test_file = f"../../datasets/{log}/{log}_test.csv"


# Compute the time differences within each CaseID
def compute_time_diffs(group):
    group = group.copy()
    group['time_diff'] = group['TimestampContact'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().astype(int)
    group['time_since_start'] = (group['TimestampContact'] - group['TimestampContact'].min()).dt.total_seconds().astype(int)
    group['remtime_std'] = group['time_since_start'].std()  # Calculate the standard deviation as a representation for remtime_std
    return group

def process_data(df):
    """Apply all preprocessing steps to a dataframe"""
    # Rename trace_id to CaseID if not already done
    if 'trace_id' in df.columns:
        df = df.rename(columns={'trace_id': 'CaseID'})
    
    # Handle timestamps
    df['TimestampContact'] = df['TimestampContact'].apply(
        lambda x: re.sub(r'(T\d{2}:\d{2})$', r'\1:00', x)
    )
    df['TimestampContact'] = pd.to_datetime(df['TimestampContact'], errors='coerce')
    df = df.dropna(subset=['TimestampContact'])
    
    # Sort and create activity column
    df = df.sort_values(['CaseID', 'TimestampContact'])
    df['Activity'] = df['topic'] + ' - ' + df['subtopic']
    
    # Compute time differences
    df = df.groupby('CaseID').apply(compute_time_diffs).reset_index(drop=True)
    
    # Aggregate per CaseID
    agg_data = df.groupby('CaseID').agg({
        'Activity': lambda x: ', '.join(x),
        'time_diff': lambda x: ', '.join(map(str, x.astype(int))),
        'time_since_start': lambda x: ', '.join(map(str, x.astype(int))),
        'topic': lambda x: ', '.join(x),
        'subtopic': lambda x: ', '.join(x),
        'remtime_std': 'mean'
    }).reset_index()
    
    # Rename columns
    agg_data.rename(columns={
        'Activity': 'trace',
        'time_diff': 'time_column1',
        'time_since_start': 'time_column2',
        'topic': 'cat_column1',
        'subtopic': 'cat_column2',
    }, inplace=True)
    
    # Process numerical columns
    agg_data['num_column1'] = agg_data['remtime_std'].apply(lambda x: [int(x)] if not pd.isna(x) else [0]) # Replace NaN with a default value like 0
    agg_data['num_column2'] = agg_data['time_column2'].apply(lambda x: [int(i) for i in x.split(', ')])
    
    return agg_data.dropna()

def compute_time_diffs(group):
    group = group.copy()
    group['time_diff'] = group['TimestampContact'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().astype(int)
    group['time_since_start'] = (group['TimestampContact'] - group['TimestampContact'].min()).dt.total_seconds().astype(int)
    group['remtime_std'] = group['time_since_start'].std()
    return group

# Load and process both datasets
train_val_df = pd.read_csv(trainval_file, encoding='latin-1')
test_df = pd.read_csv(test_file, encoding='latin-1')

# Process both datasets
processed_trainval = process_data(train_val_df)
processed_test = process_data(test_df)

# Combine processed data for the full dataset
final_columns = ['CaseID', 'trace', 'time_column1', 'time_column2', 'cat_column1', 'cat_column2', 'num_column1', 'num_column2']
final_data = pd.concat([processed_trainval[final_columns], processed_test[final_columns]], axis=0)

# Save the combined processed dataset
final_data.to_csv(f'{output_dir}/data.csv', index=False)

# Get case IDs for train/val split from processed_trainval
trainval_cases = processed_trainval['CaseID'].unique()
train_ids = trainval_cases[:int(0.824 * len(trainval_cases))]  # Adjust split to maintain ~70% of total
valid_ids = trainval_cases[int(0.824 * len(trainval_cases)):]  # Remaining ~15% of total

# Get test case IDs from processed_test
test_ids = processed_test['CaseID'].unique()

# Save the index files
pd.DataFrame(train_ids, columns=['CaseID']).to_csv(f'{output_dir}/train_index.csv', index=False)
pd.DataFrame(valid_ids, columns=['CaseID']).to_csv(f'{output_dir}/valid_index.csv', index=False)
pd.DataFrame(test_ids, columns=['CaseID']).to_csv(f'{output_dir}/test_index.csv', index=False)
