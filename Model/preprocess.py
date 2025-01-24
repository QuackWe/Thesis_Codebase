import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sys import argv
import os
import pandas as pd
from datetime import datetime
from dataloader import TraceDataset


def normalize_timestamps(df, column_name, target_format='%Y-%m-%d %H:%M:%S'):
    print(f"--- Normalizing timestamps in column: {column_name} ---")

    def convert_to_format(value):
        try:
            return datetime.fromisoformat(value).strftime(target_format)
        except ValueError:
            try:
                return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S').strftime(target_format)
            except ValueError:
                try:
                    return datetime.strptime(value, '%Y-%m-%dT%H:%M').strftime(target_format)
                except ValueError:
                    return None

    df[column_name] = df[column_name].apply(convert_to_format)
    unparseable = df[column_name].isnull().sum()
    print(f"Unparseable values after normalization: {unparseable}")
    return df


def separate_consistent_traces(df):
    # Create a mask for consistent cases
    consistent_mask = ~df['CustomerId'].isin(
        df.groupby('CustomerId')['outcome']
        .filter(lambda x: len(x.unique()) > 1)
        .index
    )

    # Split into consistent and inconsistent datasets
    consistent_df = df[consistent_mask].copy()
    inconsistent_df = df[~consistent_mask].copy()

    # Print statistics
    total_cases = df['CustomerId'].nunique()
    consistent_cases = consistent_df['CustomerId'].nunique()
    inconsistent_cases = inconsistent_df['CustomerId'].nunique()

    print(f"Total cases: {total_cases}")
    print(f"Consistent cases: {consistent_cases}")
    print(f"Inconsistent cases: {inconsistent_cases}")
    print(f"Percentage consistent: {(consistent_cases / total_cases) * 100:.2f}%")

    return consistent_df, inconsistent_df

def robust_scale(series, eps=1e-8):
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    denom = (q75 - q25)
    if abs(denom) < eps:
        denom = eps
    return (series - series.median()) / denom

def process_traces(df, log_name, is_consistent=True):
    # Combine Topic and Subtopic to create the Activity column
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)

    # Add relative time calculation with explicit format
    df['RelativeTime'] = df.groupby('CustomerId')['TimestampContact'].transform(
        lambda x: (pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S') -
                   pd.to_datetime(x.iloc[0], format='%Y-%m-%d %H:%M:%S')).dt.total_seconds() / 3600 # Converted to hours
    )

    # Add debug prints here
    print("\nInitial RelativeTime stats:")
    print(df['RelativeTime'].describe())
    print("NaN in RelativeTime:", df['RelativeTime'].isnull().sum())

    # Log transform
    df['LogTime'] = np.log1p(df['RelativeTime'])
    print("\nLogTime stats:")
    print(df['LogTime'].describe())
    print("NaN in LogTime:", df['LogTime'].isnull().sum())

    # Normalize relative times to [0,1] range within each trace
    # 1. Log transform to handle wide time ranges
    df['LogTime'] = np.log1p(df['RelativeTime'])
    # 2. Robust standardization per customer
    df['NormalizedTime'] = df.groupby('CustomerId')['LogTime'].transform(robust_scale)
    df['NormalizedTime'] = df['NormalizedTime'].clip(-5, 5)

    # 3. Handle outliers
    # df['NormalizedTime'] = df['NormalizedTime'].clip(-5, 5)
    num_nans = df['NormalizedTime'].isnull().sum()
    if num_nans > 0:
        print(f"Warning: {num_nans} NaN values remain after robust scaling and clipping.")

    df = df.sort_values(by=['CustomerId', 'TimestampContact'])

    # Group by CustomerId to create traces
    traces = df.groupby('CustomerId')['Activity'].apply(list).reset_index()
    outcomes = df.groupby('CustomerId')['outcome'].first().reset_index()
    types = df.groupby('CustomerId')['type_of_customer'].first().reset_index()
    times = df.groupby('CustomerId')['NormalizedTime'].apply(list).reset_index()

    # Create activity-to-ID mapping
    unique_activities = df['Activity'].unique()
    activity_to_id = {activity: idx for idx, activity in enumerate(unique_activities)}
    traces['ActivityIDs'] = traces['Activity'].apply(lambda x: [activity_to_id[a] for a in x])

    # Create a mapping for type_of_customer
    unique_customer_types = df['type_of_customer'].unique()
    customer_type_to_id = {cust_type: idx for idx, cust_type in enumerate(unique_customer_types)}
    types['TypeID'] = types['type_of_customer'].map(customer_type_to_id)

    # Label encode outcomes
    outcome_encoder = LabelEncoder()
    outcomes['OutcomeID'] = outcome_encoder.fit_transform(outcomes['outcome']).astype(int)

    # Create next activity labels by shifting the sequence
    traces['NextActivityIDs'] = traces['ActivityIDs'].apply(lambda x: x[1:] + [0])  # Shift and pad with 0

    # Convert traces to padded sequences
    max_seq_length = max(len(trace) for trace in traces['ActivityIDs'])
    print(f"Max sequence length determined: {max_seq_length}")

    padded_traces = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(trace[:max_seq_length]) for trace in traces['ActivityIDs']],
        batch_first=True,
        padding_value=0
    )

    padded_next_activities = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(trace[:max_seq_length]) for trace in traces['NextActivityIDs']],
        batch_first=True,
        padding_value=0
    )

    padded_times = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(time[:max_seq_length], dtype=torch.float) for time in times['NormalizedTime']],
        batch_first=True,
        padding_value=0.0
    )

    # Create attention masks
    attention_masks = torch.tensor([
        [1] * len(trace[:max_seq_length]) + [0] * (max_seq_length - len(trace))
        for trace in traces['ActivityIDs']
    ])

    # Prepare the dataset
    dataset = pd.DataFrame({
        'CustomerId': traces['CustomerId'],
        'Trace': [trace.tolist() for trace in padded_traces],
        'Times': [time.tolist() for time in padded_times],
        'NextActivity': [trace.tolist() for trace in padded_next_activities],
        'AttentionMask': attention_masks.tolist(),
        'Outcome': outcomes['OutcomeID'],
        'CustomerType': types['TypeID']
    })

    # Create PyTorch Dataset and save it
    torch_dataset = TraceDataset(dataset)
    suffix = "consistent" if is_consistent else "inconsistent"
    save_path = f"datasets/{log_name}/pytorch_dataset_{suffix}.pt"
    torch.save(torch_dataset, save_path)
    print(f"Processed dataset saved to {save_path}")

    # Save mappings for later use
    mappings = {
        'activity_to_id': activity_to_id,
        'customer_type_to_id': customer_type_to_id,
        'outcome_encoder': outcome_encoder
    }
    mapping_path = f"datasets/{log_name}/mappings_{suffix}.pt"
    torch.save(mappings, mapping_path)


if __name__ == "__main__":
    # Command-line arguments
    log = argv[1]
    working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
    input_file = f"{working_directory}/Andrei_thesis_KRIF_{log}_vPaul_v3.csv"

    # Create the output directory
    os.makedirs(f"datasets/{log}", exist_ok=True)

    # Load the input data
    df = pd.read_csv(input_file, encoding='latin-1')

    # Normalize timestamps
    df = normalize_timestamps(df, 'TimestampContact')

    # Separate consistent and inconsistent traces
    consistent_df, inconsistent_df = separate_consistent_traces(df)

    # Process both datasets separately
    process_traces(consistent_df, log, is_consistent=True)
    process_traces(inconsistent_df, log, is_consistent=False)
