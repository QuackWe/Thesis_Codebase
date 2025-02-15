import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sys import argv
import os
import gc
import pandas as pd
from datetime import datetime


# Import your local TraceDataset definition if you need it for the smaller dataset approach
# from dataloader import TraceDataset

# ----------------------
# 1) Utility Functions
# ----------------------

def get_trace_identifier(df, dataset_type):
    """
    Assigns a trace identifier column to the dataframe based on its dataset type.
    """
    if dataset_type == 'application':
        df['trace_id'] = (
                df['CustomerId'].astype(str) + '_'
                + df['trace_nr'].astype(str) + '_'
                + df['BusinessLine'].astype(str)
        )
    else:
        df['trace_id'] = df['CustomerId'].astype(str)
    return df


def calculate_global_normalized_times(df, group_col='trace_id'):
    """Calculate time features with global normalization"""
    print("\n=== Calculating global-normalized time features ===")

    # Calculate relative times within traces
    df['RelativeTime'] = df.groupby(group_col)['TimestampContact'].transform(
        lambda x: (pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S') -
                   pd.to_datetime(x.iloc[0], format='%Y-%m-%d %H:%M:%S')
                   ).dt.total_seconds() / 3600
    )

    # Calculate log time
    df['LogTime'] = np.log1p(df['RelativeTime'])

    # Global robust scaling
    df['NormalizedTime'] = robust_scale(df['LogTime'])
    df['NormalizedTime'] = df['NormalizedTime'].clip(-5, 5)

    # Debug prints
    print("Global time statistics:")
    print(f"Median LogTime: {df['LogTime'].median():.2f}")
    print(f"IQR LogTime: {df['LogTime'].quantile(0.75) - df['LogTime'].quantile(0.25):.2f}")
    print(f"NormalizedTime range: [{df['NormalizedTime'].min():.2f}, {df['NormalizedTime'].max():.2f}]")

    return df


def normalize_timestamps(df, column_name, target_format='%Y-%m-%d %H:%M:%S'):
    """
    Converts timestamps in a given column to a uniform format, applying a series of fallback parsers.
    """
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


def separate_consistent_traces(df, dataset_type):
    """
    Splits records into consistent vs inconsistent traces based on outcomes
    within each trace. If a single trace/id has multiple different outcomes, it is inconsistent.
    """
    df = get_trace_identifier(df, dataset_type)

    if dataset_type == 'application':
        consistent_mask = ~df['trace_id'].isin(
            df.groupby('trace_id')['outcome']
            .filter(lambda x: len(x.unique()) > 1)
            .index
        )
        id_col = 'trace_id'
    else:
        consistent_mask = ~df['CustomerId'].isin(
            df.groupby('CustomerId')['outcome']
            .filter(lambda x: len(x.unique()) > 1)
            .index
        )
        id_col = 'CustomerId'

    consistent_df = df[consistent_mask].copy()
    inconsistent_df = df[~consistent_mask].copy()

    # Debug prints
    total_cases = df[id_col].nunique()
    consistent_cases = consistent_df[id_col].nunique()
    inconsistent_cases = inconsistent_df[id_col].nunique()
    print(f"Total cases: {total_cases}")
    print(f"Consistent cases: {consistent_cases}")
    print(f"Inconsistent cases: {inconsistent_cases}")
    print(f"Percentage consistent: {(consistent_cases / total_cases) * 100:.2f}%")
    return consistent_df, inconsistent_df


def robust_scale(series, eps=1e-8):
    """
    Performs a robust scaling using the median and IQR.
    Clamps very small denominators to eps to avoid division by zero.
    """
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    denom = q75 - q25
    if abs(denom) < eps:
        denom = eps
    return (series - series.median()) / denom


def remove_orientation_activity_traces(df, activity_to_remove='Online_OriÃ«ntatie'):
    """
    Removes traces that only contain the specified activity.

    Args:
        df: DataFrame containing the traces
        activity_to_remove: Activity to check for single-activity traces
    """
    # Create Activity column and trace_id
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)
    df = get_trace_identifier(df, 'application')

    # Group by trace_id to get all activities in each trace
    trace_activities = df.groupby('trace_id')['Activity'].unique()

    # Find traces that only contain the specified activity
    single_activity_traces = trace_activities[
        trace_activities.apply(lambda x: len(x) == 1 and x[0] == activity_to_remove)
    ].index

    # Remove these traces from the dataset
    clean_df = df[~df['trace_id'].isin(single_activity_traces)]

    print(f"Removed {len(single_activity_traces)} traces containing only '{activity_to_remove}'")
    print(f"Dataset size before: {len(df)}, after: {len(clean_df)}")

    return clean_df


def remove_adobe_aanvraag_activities(df):
    """
    Removes 'Aanvraag_Aanvraag gestart' activities where source is Adobe

    Args:
        df: DataFrame containing the activity data
    """
    # Create Activity column
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)

    # Create mask for rows to remove
    mask = ~((df['Activity'] == 'Aanvraag_Aanvraag gestart') &
             (df['Source'] == 'Adobe'))

    # Remove matching rows
    clean_df = df[mask]

    # Print statistics
    removed_count = len(df) - len(clean_df)
    print(f"Removed {removed_count} 'Aanvraag_Aanvraag gestart' activities with Adobe source")
    print(f"Dataset size before: {len(df)}, after: {len(clean_df)}")

    return clean_df


def remove_contractwijziging_activities(df):
    """
    Removes 'Contractwijziging_Periodieke overboeking' activities from the dataset

    Args:
        df: DataFrame containing the activity data
    """
    # Create Activity column
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)

    # Create mask for rows to keep
    mask = df['Activity'] != 'Contractwijziging_Periodieke overboeking'

    # Remove matching rows
    clean_df = df[mask]

    # Print statistics
    removed_count = len(df) - len(clean_df)
    print(f"Removed {removed_count} 'Contractwijziging_Periodieke overboeking' activities")
    print(f"Dataset size before: {len(df)}, after: {len(clean_df)}")

    return clean_df


def remove_single_activity_traces(df, dataset_type):
    """
    Removes all traces that contain only one activity.

    Args:
        df: DataFrame containing the traces
        dataset_type: Type of dataset ('application' or 'mortgages')
    """
    # Create Activity column and trace_id
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)
    df = get_trace_identifier(df, dataset_type)

    # Group by trace identifier to get trace lengths
    group_col = 'trace_id' if dataset_type == 'application' else 'CustomerId'
    trace_lengths = df.groupby(group_col)['Activity'].count()

    # Find traces with more than one activity
    valid_traces = trace_lengths[trace_lengths > 1].index

    # Keep only traces with more than one activity
    clean_df = df[df[group_col].isin(valid_traces)]

    print(f"Removed {len(trace_lengths) - len(valid_traces)} single-activity traces")
    print(f"Dataset size before: {len(df)}, after: {len(clean_df)}")

    return clean_df


def merge_funnel_lead_activities(df):
    """
    Merges all activities where topic is 'Funnel' and subtopic starts with 'Lead'
    into a single 'Funnel_Lead' activity.

    Args:
        df: DataFrame containing the activity data
    """
    # Store original count
    original_count = len(df['Activity'].unique())

    # Create mask for Funnel_Lead activities
    funnel_lead_mask = (df['topic'] == 'Funnel') & (df['subtopic'].str.startswith('Lead'))

    # Print some statistics before merging
    funnel_lead_counts = df[funnel_lead_mask].groupby(['topic', 'subtopic']).size()
    print("\nFunnel_Lead activities before merging:")
    print(funnel_lead_counts)

    # Modify subtopic for matching rows
    df.loc[funnel_lead_mask, 'subtopic'] = 'Lead'

    # Create new Activity column
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)

    print(f"\nMerged all Funnel_Lead variants into single 'Funnel_Lead' activity")
    print(f"Dataset size before: {original_count}, after: {len(df['Activity'].unique())}")

    return df


def reduce_orientation_events(df):
    """
    Reduces consecutive Online_Oriëntatie events to first and last in each sequence.
    Maintains all original columns and matches the style of other cleaning functions.
    """
    print("\n=== Reducing consecutive orientation events ===")
    original_size = len(df)
    print(f"Dataset size before orientation reduction: {original_size:,}")

    # Create trace identifier if not exists
    if 'trace_id' not in df.columns:
        df = get_trace_identifier(df, 'application')

    # Sort by trace and timestamp
    df = df.sort_values(['trace_id', 'TimestampContact'])

    # Process traces in groups
    reduced_dfs = []
    for trace_id, group in df.groupby('trace_id', sort=False):
        activities = group['Activity'].tolist()
        reduced_indices = []

        i = 0
        while i < len(activities):
            if activities[i] == 'Online_OriÃ«ntatie':
                start_idx = i
                while i < len(activities) and activities[i] == 'Online_OriÃ«ntatie':
                    i += 1
                # Keep first and last if sequence length > 1
                reduced_indices.append(start_idx)
                if i - start_idx > 1:
                    reduced_indices.append(i - 1)
            else:
                reduced_indices.append(i)
                i += 1

        # Preserve all original columns
        reduced_dfs.append(group.iloc[reduced_indices])

    # Combine results
    reduced_df = pd.concat(reduced_dfs, ignore_index=True)

    # Calculate and print statistics
    removed_count = original_size - len(reduced_df)
    print(f"Removed {removed_count} redundant orientation events")
    print(f"Dataset size after orientation reduction: {len(reduced_df):,}")
    print("=" * 50)

    return reduced_df


# --------------------------------------------------------------------
# 2) Smaller "process_traces" function for mortgages & single-run sets
# --------------------------------------------------------------------

def process_traces(df, log_name, is_consistent=True):
    """
    Processes relatively smaller datasets (e.g. mortgages),
    generates a final Pytorch dataset directly in memory.
    """
    df = get_trace_identifier(df, log_name)

    # Create an "Activity" field
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)

    # Calculate time features
    # For mortgages, we group by 'CustomerId'
    # For application, we group by 'trace_id' (handled below if needed)
    df['RelativeTime'] = df.groupby('CustomerId')['TimestampContact'].transform(
        lambda x: (
                          pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
                          - pd.to_datetime(x.iloc[0], format='%Y-%m-%d %H:%M:%S')
                  ).dt.total_seconds() / 3600
    )

    print("\nInitial RelativeTime stats:")
    print(df['RelativeTime'].describe())
    print("NaN in RelativeTime:", df['RelativeTime'].isnull().sum())

    df['LogTime'] = np.log1p(df['RelativeTime'])
    print("\nLogTime stats:")
    print(df['LogTime'].describe())
    print("NaN in LogTime:", df['LogTime'].isnull().sum())

    df['NormalizedTime'] = df.groupby('CustomerId')['LogTime'].transform(robust_scale)
    # df['NormalizedTime'] = df['NormalizedTime'].clip(-5, 5)

    nan_count = df['NormalizedTime'].isnull().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values remain after robust scaling and clipping.")

    # Sort for mortgages
    if log_name == 'mortgages':
        df = df.sort_values(by=['CustomerId', 'TimestampContact'])
        groupby_col = 'CustomerId'
    else:
        # For application usage in this function, fallback
        df = df.sort_values(by=['trace_id', 'TimestampContact'])
        groupby_col = 'trace_id'

    # Group into traces
    traces = df.groupby(groupby_col)['Activity'].apply(list).reset_index()
    outcomes = df.groupby(groupby_col)['outcome'].first().reset_index()
    types = df.groupby(groupby_col)['type_of_customer'].first().reset_index()
    times = df.groupby(groupby_col)['NormalizedTime'].apply(list).reset_index()

    # Activity -> ID mapping
    unique_activities = df['Activity'].unique()
    activity_to_id = {act: idx + 1 for idx, act in enumerate(unique_activities)}
    traces['ActivityIDs'] = traces['Activity'].apply(lambda x: [activity_to_id[a] for a in x])

    # Debug: Check if any activity is encoded as 0
    if 0 in activity_to_id.values():
        zero_activity = [act for act, idx in activity_to_id.items() if idx == 0]
        print(f"Warning: The following activity is encoded as 0: {zero_activity}")

    # Customer type -> ID mapping
    unique_customer_types = df['type_of_customer'].unique()
    customer_type_to_id = {cust_type: idx for idx, cust_type in enumerate(unique_customer_types)}
    types['TypeID'] = types['type_of_customer'].map(customer_type_to_id)

    # Encode outcomes
    outcome_encoder = LabelEncoder()
    outcomes['OutcomeID'] = outcome_encoder.fit_transform(outcomes['outcome']).astype(int)

    # Next-activity
    traces['NextActivityIDs'] = traces['ActivityIDs'].apply(lambda seq: seq[1:] + [0])

    max_seq_length = max(len(trace) for trace in traces['ActivityIDs'])
    print(f"Max sequence length determined: {max_seq_length}")

    # Padded sequences
    padded_traces = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(trace[:max_seq_length]) for trace in traces['ActivityIDs']],
        batch_first=True,
        padding_value=0
    )
    padded_next = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(trace[:max_seq_length]) for trace in traces['NextActivityIDs']],
        batch_first=True,
        padding_value=0
    )
    padded_times = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(tt[:max_seq_length], dtype=torch.float) for tt in times['NormalizedTime']],
        batch_first=True,
        padding_value=0
    )
    attention_masks = torch.tensor([
        [1] * len(trace[:max_seq_length]) + [0] * (max_seq_length - len(trace))
        for trace in traces['ActivityIDs']
    ])

    # Build final DataFrame
    if log_name == 'mortgages':
        dataset = pd.DataFrame({
            'CustomerId': traces['CustomerId'],
            'Trace': [t.tolist() for t in padded_traces],
            'Times': [t.tolist() for t in padded_times],
            'NextActivity': [t.tolist() for t in padded_next],
            'AttentionMask': attention_masks.tolist(),
            'Outcome': outcomes['OutcomeID'],
            'CustomerType': types['TypeID']
        })
    else:
        dataset = pd.DataFrame({
            'CustomerId': traces['trace_id'],
            'Trace': [t.tolist() for t in padded_traces],
            'Times': [t.tolist() for t in padded_times],
            'NextActivity': [t.tolist() for t in padded_next],
            'AttentionMask': attention_masks.tolist(),
            'Outcome': outcomes['OutcomeID'],
            'CustomerType': types['TypeID']
        })

    # Wrap in a standard Torch dataset (optional local definition)
    from dataloader import TraceDataset
    torch_dataset = TraceDataset(dataset)

    suffix = "consistent" if is_consistent else "inconsistent"
    save_path = f"datasets/{log_name}/pytorch_dataset_{suffix}.pt"
    torch.save(torch_dataset, save_path)
    print(f"Processed dataset saved to {save_path}")

    # Save mappings
    mappings = {
        'activity_to_id': activity_to_id,
        'customer_type_to_id': customer_type_to_id,
        'outcome_encoder': outcome_encoder
    }
    mapping_path = f"datasets/{log_name}/mappings_{suffix}.pt"
    torch.save(mappings, mapping_path)
    print(f"Mappings saved to {mapping_path}")


# -------------------------------------------------------------------------
# 3) Large Dataset Function: Memory-Mapped / Batching for 'application'
# -------------------------------------------------------------------------

# Define a re-creatable lazy-loading dataset
class MemoryMappedDataset(Dataset):
    def __init__(self, data_dir):
        # FutureWarning can be silenced or addressed by setting weights_only=True if all data is numeric
        self.meta = torch.load(f"{data_dir}/metadata.pt", weights_only=False)
        self.length = self.meta['length']
        self.max_seq_len = self.meta['max_seq_length']

        self.trace_mmap = np.memmap(
            f"{data_dir}/Trace.dat", dtype=np.float32, mode='r',
            shape=(self.length, self.max_seq_len)
        )
        self.next_mmap = np.memmap(
            f"{data_dir}/NextActivity.dat", dtype=np.float32, mode='r',
            shape=(self.length, self.max_seq_len)
        )
        self.times_mmap = np.memmap(
            f"{data_dir}/Times.dat", dtype=np.float32, mode='r',
            shape=(self.length, self.max_seq_len)
        )
        self.mask_mmap = np.memmap(
            f"{data_dir}/AttentionMask.dat", dtype=np.uint8, mode='r',
            shape=(self.length, self.max_seq_len)
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'trace': torch.tensor(self.trace_mmap[idx], dtype=torch.float32),
            'next_activity': torch.tensor(self.next_mmap[idx], dtype=torch.long),
            'times': torch.tensor(self.times_mmap[idx], dtype=torch.float32),
            'mask': torch.tensor(self.mask_mmap[idx], dtype=torch.long),
            'outcome': torch.tensor(self.meta['Outcome'][idx], dtype=torch.long),
            'customer_type': torch.tensor(self.meta['CustomerType'][idx], dtype=torch.long),
            'customer_id_code': torch.tensor(self.meta['CustomerIdCodes'][idx], dtype=torch.long)
        }


def process_traces_in_batches(df, log_name, batch_size=1000, is_consistent=True):
    """
    Handles large application datasets using memory-mapped files
    to avoid excessive RAM usage.
    """
    df = get_trace_identifier(df, log_name)
    group_col = 'trace_id' if log_name == 'application' else 'CustomerId'

    # Create Activity
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)

    # # Time features
    # df['RelativeTime'] = df.groupby(group_col)['TimestampContact'].transform(
    #     lambda x: (
    #                       pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
    #                       - pd.to_datetime(x.iloc[0], format='%Y-%m-%d %H:%M:%S')
    #               ).dt.total_seconds() / 3600
    # )
    # df['LogTime'] = np.log1p(df['RelativeTime'])
    # df['NormalizedTime'] = df.groupby(group_col)['LogTime'].transform(robust_scale)
    # df['NormalizedTime'] = df['NormalizedTime'].clip(-5, 5)
    #
    # # Debug prints
    # print("\nInitial RelativeTime stats:")
    # print(df['RelativeTime'].describe())
    # print("NaN in RelativeTime:", df['RelativeTime'].isnull().sum())
    #
    # print("\nLogTime stats:")
    # print(df['LogTime'].describe())
    # print("NaN in LogTime:", df['LogTime'].isnull().sum())

    n_nans = df['NormalizedTime'].isnull().sum()
    if n_nans > 0:
        print(f"Warning: {n_nans} NaN values remain after robust scaling and clipping.")

    # Batch grouping
    unique_ids = df[group_col].unique()
    all_traces, all_outcomes, all_types, all_times = [], [], [], []

    for i in range(0, len(unique_ids), batch_size):
        bid = unique_ids[i:i + batch_size]
        batch_df = df[df[group_col].isin(bid)].copy()

        batch_traces = batch_df.groupby(group_col)['Activity'].apply(list).reset_index()
        batch_outcomes = batch_df.groupby(group_col)['outcome'].first().reset_index()
        batch_types = batch_df.groupby(group_col)['type_of_customer'].first().reset_index()
        batch_times = batch_df.groupby(group_col)['NormalizedTime'].apply(list).reset_index()

        all_traces.append(batch_traces)
        all_outcomes.append(batch_outcomes)
        all_types.append(batch_types)
        all_times.append(batch_times)

        del batch_df
        gc.collect()

    traces = pd.concat(all_traces, ignore_index=True)
    outcomes = pd.concat(all_outcomes, ignore_index=True)
    types = pd.concat(all_types, ignore_index=True)
    times = pd.concat(all_times, ignore_index=True)

    # Activity -> ID
    unique_activities = df['Activity'].unique()
    activity_to_id = {act: idx + 1 for idx, act in enumerate(unique_activities)}
    traces['ActivityIDs'] = traces['Activity'].apply(lambda x: [activity_to_id[a] for a in x])

    # Next Activities
    traces['NextActivityIDs'] = traces['ActivityIDs'].apply(lambda seq: seq[1:] + [0])

    # Customer Type -> ID
    unique_customer_types = df['type_of_customer'].unique()
    customer_type_to_id = {c: idx for idx, c in enumerate(unique_customer_types)}
    types['TypeID'] = types['type_of_customer'].map(customer_type_to_id)

    # Outcomes -> ID
    outcome_encoder = LabelEncoder()
    outcomes['OutcomeID'] = outcome_encoder.fit_transform(outcomes['outcome']).astype(int)

    # Determine max sequence length
    print("Calculating global maximum sequence length...")
    max_seq_length = max(len(x) for x in traces['ActivityIDs'])
    print(f"Global max sequence length: {max_seq_length}")

    # Cleanup original df
    del df
    gc.collect()

    output_dir = f"datasets/{log_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup memory-mapped arrays
    n_cases = len(traces)
    components = {
        'Trace': (n_cases, max_seq_length),
        'Times': (n_cases, max_seq_length),
        'NextActivity': (n_cases, max_seq_length),
        'AttentionMask': (n_cases, max_seq_length)
    }

    mmap_files = {
        name: np.memmap(
            f"{output_dir}/{name}.dat",
            dtype=np.float32 if name != 'AttentionMask' else np.uint8,
            mode='w+',
            shape=shape
        )
        for name, shape in components.items()
    }

    current_idx = 0
    micro_batch_size = 100

    # Micro-batch write
    for start_idx in range(0, n_cases, micro_batch_size):
        batch_traces = traces['ActivityIDs'].iloc[start_idx:start_idx + micro_batch_size]
        batch_next = traces['NextActivityIDs'].iloc[start_idx:start_idx + micro_batch_size]
        batch_times = times['NormalizedTime'].iloc[start_idx:start_idx + micro_batch_size]

        # Allocate
        padded_traces = np.zeros((len(batch_traces), max_seq_length), dtype=np.float32)
        padded_next = np.zeros((len(batch_next), max_seq_length), dtype=np.float32)
        padded_times = np.zeros((len(batch_times), max_seq_length), dtype=np.float32)
        masks = np.zeros((len(batch_traces), max_seq_length), dtype=np.uint8)

        for j, seq in enumerate(batch_traces):
            l = min(len(seq), max_seq_length)
            padded_traces[j, :l] = seq[:l]
            masks[j, :l] = 1

        for j, seq in enumerate(batch_next):
            l = min(len(seq), max_seq_length)
            padded_next[j, :l] = seq[:l]

        for j, tvals in enumerate(batch_times):
            l = min(len(tvals), max_seq_length)
            padded_times[j, :l] = tvals[:l]

        mmap_files['Trace'][current_idx:current_idx + len(batch_traces)] = padded_traces
        mmap_files['NextActivity'][current_idx:current_idx + len(batch_traces)] = padded_next
        mmap_files['Times'][current_idx:current_idx + len(batch_traces)] = padded_times
        mmap_files['AttentionMask'][current_idx:current_idx + len(batch_traces)] = masks

        current_idx += len(batch_traces)
        del (batch_traces, batch_next, batch_times,
             padded_traces, padded_next, padded_times, masks)
        gc.collect()

    # Save metadata with minimal references
    cust_ids = traces[group_col].astype('category')
    metadata = {
        'CustomerIdCodes': cust_ids.cat.codes.to_numpy(dtype=np.int32),
        'CustomerIdCategories': list(cust_ids.cat.categories),
        'Outcome': outcomes['OutcomeID'].to_numpy(dtype=np.int32),
        'CustomerType': types['TypeID'].to_numpy(dtype=np.int32),
        'activity_to_id': activity_to_id,
        'customer_type_to_id': customer_type_to_id,
        'outcome_classes': list(outcome_encoder.classes_),
        'length': n_cases,
        'max_seq_length': max_seq_length
    }
    torch.save(metadata, f"{output_dir}/metadata.pt")
    print("Memory-mapped arrays created successfully. Metadata saved.")

    # Rebuild an outcome_encoder using the stored classes
    outcome_encoder_for_app = LabelEncoder()
    outcome_encoder_for_app.classes_ = np.array(metadata['outcome_classes'], dtype=object)

    # Create the mappings dict
    mappings = {
        'activity_to_id': metadata['activity_to_id'],
        'customer_type_to_id': metadata['customer_type_to_id'],
        'outcome_encoder': outcome_encoder_for_app
    }

    torch.save(mappings, f"{output_dir}/mappings_consistent.pt")
    print(
        "For large 'application' dataset, created mappings_consistent.pt to match the training script’s expectations.")


    # Save only a small dictionary referencing the folder
    dataset_info = {
        'dataset_dir': output_dir,
        'is_consistent': is_consistent
    }
    suffix = "consistent" if is_consistent else "inconsistent"
    torch.save(dataset_info, f"{output_dir}/pytorch_dataset_{suffix}.pt")
    print(f"Dataset info file saved at {output_dir}/pytorch_dataset_{suffix}.pt\n"
          "This file is intentionally small and references the large .dat files on disk.")


# ---------------
# 4) Main driver
# ---------------

if __name__ == "__main__":
    log = argv[1]
    working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
    input_file = f"{working_directory}/Andrei_thesis_KRIF_{log}_v3.csv"

    # Create the output directory
    os.makedirs(f"datasets/{log}", exist_ok=True)

    if log == 'mortgages':
        df = pd.read_csv(input_file, encoding='latin-1')
        df = normalize_timestamps(df, 'TimestampContact')
        df = get_trace_identifier(df, 'mortgages')

        # Calculate time features FIRST
        df = calculate_global_normalized_times(df)
        # Remove single-activity traces
        df = remove_single_activity_traces(df, log)
        # Merge Funnel_Lead activities before other processing
        df = merge_funnel_lead_activities(df)
        # Split consistent & inconsistent
        consistent_df, inconsistent_df = separate_consistent_traces(df, log)
        process_traces(consistent_df, log, is_consistent=True)
        process_traces(inconsistent_df, log, is_consistent=False)

    elif log == 'application':
        # Large dataset approach
        df = pd.read_csv(input_file, encoding='latin-1', sep='|')
        df = normalize_timestamps(df, 'TimestampContact')
        df = get_trace_identifier(df, 'application')

        # Calculate time features FIRST
        df = calculate_global_normalized_times(df)

         # Remove traces where only orientation browsing happens
        df = remove_orientation_activity_traces(df)
        # Remove Adobe Aanvraag activities before processing
        df = remove_adobe_aanvraag_activities(df)
        # Remove Contractwijziging activities before processing
        df = remove_contractwijziging_activities(df)
        # Merge online orinetatie events
        df = reduce_orientation_events(df)
        # Remove single-activity traces
        df = remove_single_activity_traces(df, log)

        batch_size = 100  # Adjust as needed
        process_traces_in_batches(df, log, batch_size=batch_size, is_consistent=True)
