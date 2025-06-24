import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pm4py.objects.log.importer.xes import importer as xes_importer

# ------------------------------
# Util Functions
# ------------------------------

def remove_orientation_activity_traces(df, activity_to_remove='Online_OriÃƒÂ«ntatie'):
    """
    Removes traces that only contain the specified activity.

    Args:
        df: DataFrame containing the traces
        activity_to_remove: Activity to check for single-activity traces
    """
    # Create Activity column and trace_id
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)

    # Group by trace_id to get all activities in each trace
    trace_activities = df.groupby('trace_id')['Activity'].unique()

    # Find traces that only contain the specified activity
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


def remove_single_activity_traces(df):
    """
    Removes all traces that contain only one activity.

    Args:
        df: DataFrame containing the traces
        dataset_type: Type of dataset ('application' or 'mortgages')
    """
    # Create Activity column and trace_id
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)

    # Group by trace identifier to get trace lengths
    trace_lengths = df.groupby('trace_id')['Activity'].count()

    # Find traces with more than one activity
    valid_traces = trace_lengths[trace_lengths > 1].index

    # Keep only traces with more than one activity
    clean_df = df[df['trace_id'].isin(valid_traces)]

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
    Reduces consecutive Online_OriÃ«ntatie events to first and last in each sequence.
    Maintains all original columns and matches the style of other cleaning functions.
    """
    print("\n=== Reducing consecutive orientation events ===")
    original_size = len(df)
    print(f"Dataset size before orientation reduction: {original_size:,}")

    # Sort by trace and timestamp
    df = df.sort_values(['trace_id', 'TimestampContact'])

    # Process traces in groups
    reduced_dfs = []
    for trace_id, group in df.groupby('trace_id', sort=False):
        activities = group['Activity'].tolist()
        reduced_indices = []

        i = 0
        while i < len(activities):
            if activities[i] == 'Online_OriÃƒÂ«ntatie':
                start_idx = i
                while i < len(activities) and activities[i] == 'Online_OriÃƒÂ«ntatie':
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


def preprocess_bpic2017(xes_path):
    """Preprocess BPIC 2017 dataset to match existing pipeline format"""
    print(f"Processing BPIC 2017 dataset: {xes_path}")

    # Create output directory
    output_dir = "bpic2017"
    os.makedirs(output_dir, exist_ok=True)

    # Load and convert to DataFrame
    df = pd.DataFrame([
        {**event, **{'case:' + k: v for k, v in trace.attributes.items()}}
        for trace in xes_importer.apply(xes_path)
        for event in trace
    ])

    # Create required columns to match existing pipeline
    df['trace_id'] = df['case:concept:name']

    # Map EventOrigin as topic and Action as subtopic
    df['topic'] = df['EventOrigin']
    df['subtopic'] = df['Action']

    # --- Critical Fix: Handle microsecond-precision timestamps ---
    # Convert timestamps to ISO format with microseconds and timezone
    df['TimestampContact'] = pd.to_datetime(
        df['time:timestamp'],
        utc=True
    ).dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    # Convert with full precision first
    df['TimestampContact'] = pd.to_datetime(
        df['time:timestamp'],
        format='%Y-%m-%d %H:%M:%S.%f%z',  # Add %z for timezone if needed
        errors='coerce'
    )

    # Validate timestamp conversion
    nan_timestamps = df['TimestampContact'].isna().sum()
    if nan_timestamps > 0:
        print(f"Warning: {nan_timestamps} timestamps could not be parsed")

    # Format back to string with full precision
    df['TimestampContact'] = df['TimestampContact'].dt.strftime('%Y-%m-%d %H:%M:%S.%f%z')

    # --- BPIC-specific Outcome Handling ---
    df['outcome'] = df.groupby('trace_id')['Accepted'].transform('last')
    df['outcome'] = np.where(
        df['outcome'].isin([True, 'True', 'true']),
        'success',
        np.where(
            df['outcome'].isin([False, 'False', 'false']),
            'reject',
            'unknown'  # Handle missing/ambiguous values
        )
    )
    # Add after timestamp conversion
    microsecond_check = df['TimestampContact'].str.contains(r'\.\d{6}').mean()
    print(f"Microsecond preservation: {microsecond_check * 100:.1f}% of timestamps")

    tz_check = df['TimestampContact'].str.contains(r'\+00:00$').mean()
    print(f"Timezone preservation: {tz_check * 100:.1f}% contain UTC offset")

    df['type_of_customer'] = df['case:ApplicationType']

    # Keep only necessary columns
    df = df[[
        'trace_id',
        'topic',
        'subtopic',
        'outcome',
        'type_of_customer',
        'TimestampContact'
    ]]

    print(f"Processed dataset shape: {df.shape}")
    return df


def plot_prefix_length_distribution(df, dataset_name):
    """Plot the distribution of trace lengths in the dataset."""
    # Calculate trace lengths
    trace_lengths = df.groupby('trace_id').size()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    sns.histplot(data=trace_lengths, bins=range(1, trace_lengths.max() + 2), 
                discrete=True, shrink=0.8)
    
    # Customize plot
    plt.title(f'Prefix Length Distribution - {dataset_name} dataset')
    plt.xlabel('Prefix Length')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add exact counts as text above each bar
    for i in range(1, trace_lengths.max() + 1):
        count = (trace_lengths == i).sum()
        if count > 0:  # Only add text if there are traces of this length
            plt.text(i, count, str(count), 
                    horizontalalignment='center',
                    verticalalignment='bottom')
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{dataset_name}_prefix_distribution.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print statistics
    print(f"\nPrefix Length Statistics for {dataset_name}:")
    print(f"Mean prefix length: {trace_lengths.mean():.2f}")
    print(f"Median prefix length: {trace_lengths.median():.2f}")
    print(f"Min prefix length: {trace_lengths.min()}")
    print(f"Max prefix length: {trace_lengths.max()}")
    print(f"Total number of traces: {len(trace_lengths)}")

# ------------------------------
# Main Functions
# ------------------------------

def load_dataset(dataset_name):
    """Load a specific dataset."""
    print(f"\nLoading {dataset_name} dataset...")
    if dataset_name == "mortgages":
        full_df = pd.read_csv(f"{dataset_name}/Andrei_thesis_KRIF_mortgages_v3.csv")
        full_df['Activity'] = full_df['topic'].astype(str) + "_" + full_df['subtopic'].astype(str)
        full_df['trace_id'] = full_df['CustomerId'].astype(str)
    elif dataset_name == "application":
        full_df = pd.read_csv(f"{dataset_name}/Andrei_thesis_KRIF_application_v3.csv", sep='|')
        full_df['Activity'] = full_df['topic'].astype(str) + "_" + full_df['subtopic'].astype(str)
        full_df['trace_id'] = (
                full_df['CustomerId'].astype(str) + '_'
                + full_df['trace_nr'].astype(str) + '_'
                + full_df['BusinessLine'].astype(str)
        )
        # Fill missing values in outcome column with 'Transit'
        full_df['outcome'] = full_df['outcome'].fillna('Transit')
    elif dataset_name == "bpic2017":
        full_df = preprocess_bpic2017('bpic2017/BPI_Challenge_2017.xes')
        full_df['Activity'] = full_df['topic'].astype(str) + "_" + full_df['subtopic'].astype(str)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Print dataset statistics
    total_events = len(full_df)
    total_traces = full_df['trace_id'].nunique()
    unique_activities = full_df['Activity'].nunique()
    
    print("\nDataset Statistics:")
    print(f"Total events: {total_events:,}")
    print(f"Total traces: {total_traces:,}")
    print(f"Unique activities: {unique_activities}")
    print("Average events per trace: {:.2f}".format(total_events / total_traces))
    print("Max events in a trace: {}".format(full_df.groupby('trace_id').size().max()))
    print("-" * 50)

    # Plot prefix length distribution
    plot_prefix_length_distribution(full_df, dataset_name)
    
    return full_df

def prepare_stratification_labels(df, case_id_col='trace_id', activity_col='Activity'):
    """Create stratification labels based on trace length and activity composition."""
    
    # Get trace lengths and bin them
    trace_lengths = df.groupby(case_id_col).size()
    
    # Get activity proportions for each trace
    trace_activities = df.groupby(case_id_col)[activity_col].value_counts(normalize=True).unstack(fill_value=0)
    
    # Create length bins (smaller bins for better stratification)
    def get_length_bin(length):
        if length <= 5:
            return f"very_short_{length}"
        elif length <= 10:
            return f"short_{length//2*2}"
        elif length <= 20:
            return f"medium_{length//5*5}"
        else:
            return f"long_{length//10*10}"
    
    # Get most frequent activity per trace
    dominant_activities = trace_activities.idxmax(axis=1)
    
    # Combine length bin and dominant activity for stratification
    strat_labels = []
    for case_id in df[case_id_col].unique():
        length = trace_lengths[case_id]
        length_bin = get_length_bin(length)
        dom_activity = dominant_activities[case_id]
        strat_label = f"{length_bin}_{dom_activity}"
        strat_labels.append(strat_label)
    
    return pd.Series(strat_labels, index=df[case_id_col].unique())

def split_dataset(df, test_size=0.2, random_state=42, case_id_col='trace_id', activity_col='Activity'):
    """Split dataset while maintaining trace integrity and activity distributions."""
    
    # Get stratification labels
    strat_labels = prepare_stratification_labels(df, case_id_col, activity_col)
    
    # Count samples per stratum
    stratum_counts = strat_labels.value_counts()
    
    # Filter out strata with too few samples
    min_samples = max(2, int(1/test_size))  # Ensure at least 2 samples per stratum
    valid_strata = stratum_counts[stratum_counts >= min_samples].index
    
    # Keep only traces in valid strata
    valid_mask = strat_labels.isin(valid_strata)
    valid_cases = strat_labels[valid_mask].index
    valid_labels = strat_labels[valid_mask]
    
    # Use StratifiedKFold with adjusted number of splits
    n_splits = int(1/test_size)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Split the data
    train_idx, test_idx = next(skf.split(valid_cases, valid_labels))
    
    # Get case IDs for each split
    train_cases = valid_cases[train_idx]
    test_cases = valid_cases[test_idx]
    
    # Handle traces that couldn't be stratified
    unstrat_cases = set(df[case_id_col].unique()) - set(valid_cases)
    if unstrat_cases:
        print(f"Note: {len(unstrat_cases)} traces couldn't be stratified (added to train)")
        train_cases = np.append(train_cases, list(unstrat_cases))
    
    # Create the splits
    train_df = df[df[case_id_col].isin(train_cases)]
    test_df = df[df[case_id_col].isin(test_cases)]
    
    return train_df, test_df


def analyze_split_distributions(dataset_name):
    """
    Analyze and compare activity distributions across prefixes in train and test sets.
    Also check for activities and outcomes that appear only in test set.
    
    Args:
        dataset_name: Name of the dataset to analyze
    """
    # Load train and test sets
    train_df = pd.read_csv(f"{dataset_name}/{dataset_name}_train-val.csv")
    test_df = pd.read_csv(f"{dataset_name}/{dataset_name}_test.csv")
    
    # Check for activities only in test set
    train_activities = set(train_df['Activity'].unique())
    test_activities = set(test_df['Activity'].unique())
    test_only_activities = test_activities - train_activities
    
    if test_only_activities:
        print("\nWARNING: Activities that appear only in test set:")
        for act in sorted(test_only_activities):
            count = test_df[test_df['Activity'] == act].shape[0]
            traces = test_df[test_df['Activity'] == act]['trace_id'].nunique()
            print(f"- {act} (appears {count} times in {traces} traces)")
    else:
        print("\nNo activities appear exclusively in the test set")
    
    # Check for outcomes only in test set (if outcome column exists)
    if 'outcome' in train_df.columns and 'outcome' in test_df.columns:
        train_outcomes = set(train_df['outcome'].unique())
        test_outcomes = set(test_df['outcome'].unique())
        test_only_outcomes = test_outcomes - train_outcomes
        
        if test_only_outcomes:
            print("\nWARNING: Outcomes that appear only in test set:")
            for outcome in sorted(test_only_outcomes):
                count = test_df[test_df['outcome'] == outcome].shape[0]
                traces = test_df[test_df['outcome'] == outcome]['trace_id'].nunique()
                print(f"- {outcome} (appears in {traces} traces, {count} events)")
        else:
            print("\nNo outcomes appear exclusively in the test set")
    
    # Calculate prefix lengths for each trace
    def get_prefix_distributions(df):
        # Group by trace and get prefix lengths
        prefix_lengths = df.groupby('trace_id').size()
        # Get activity distributions for each prefix length
        prefix_dist = {}
        prefix_traces = {}  # Store trace IDs for each prefix length
        
        for length in range(1, prefix_lengths.max() + 1):
            # Group traces by prefix length
            traces_with_length = prefix_lengths[prefix_lengths == length].index
            if len(traces_with_length) > 0:
                activities = df[df['trace_id'].isin(traces_with_length)]['Activity'].value_counts(normalize=True)
                prefix_dist[length] = activities
                prefix_traces[length] = traces_with_length
            
        return prefix_dist, prefix_traces
    
    # Get distributions and trace mappings
    train_dist, train_traces = get_prefix_distributions(train_df)
    test_dist, test_traces = get_prefix_distributions(test_df)
    
    # Print comparison
    print(f"\nDistribution Analysis for {dataset_name}")
    print("=" * 80)
    
    for prefix_len in sorted(set(train_dist.keys()) | set(test_dist.keys())):
        print(f"\nPrefix Length {prefix_len}:")
        print("-" * 40)
        
        # Get activities for this prefix length
        train_acts = train_dist.get(prefix_len, pd.Series())
        test_acts = test_dist.get(prefix_len, pd.Series())
        
        # Combine all activities
        all_activities = sorted(set(train_acts.index) | set(test_acts.index))
        
        # Print comparison table
        print(f"{'Activity':<40} {'Train %':>10} {'Test %':>10} {'Diff':>10}")
        print("-" * 72)
        
        for activity in all_activities:
            train_pct = train_acts.get(activity, 0) * 100
            test_pct = test_acts.get(activity, 0) * 100
            diff = abs(train_pct - test_pct)
            
            print(f"{activity[:40]:<40} {train_pct:>10.2f} {test_pct:>10.2f} {diff:>10.2f}")
        
        # Print summary statistics
        print("\nSummary:")
        print(f"Total traces with length {prefix_len}:")
        train_count = len(train_traces.get(prefix_len, []))
        test_count = len(test_traces.get(prefix_len, []))
        print(f"Train: {train_count}, Test: {test_count}")

def process_all_datasets():
    """Process all datasets and create train/test splits."""
    datasets = ["mortgages", "application", "bpic2017"]
    results = {}
    
    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        
        # Load dataset
        df = load_dataset(dataset_name)

        # Data cleaning
        if dataset_name == 'mortgages':
            df = remove_single_activity_traces(df)
            df = merge_funnel_lead_activities(df)
        elif dataset_name == 'application':
            df = remove_orientation_activity_traces(df)
            df = remove_adobe_aanvraag_activities(df)
            df = remove_contractwijziging_activities(df)
            df = reduce_orientation_events(df)
            df = remove_single_activity_traces(df)
        
        # Split dataset
        train_df, test_df = split_dataset(df)
        
        # Store results
        results[dataset_name] = {
            'train': train_df,
            'test': test_df
        }
        
        # Save splits to CSV
        os.makedirs(dataset_name, exist_ok=True)
        train_df.to_csv(f"{dataset_name}/{dataset_name}_train-val.csv", index=False)
        test_df.to_csv(f"{dataset_name}/{dataset_name}_test.csv", index=False)
        
        # Print split statistics
        print(f"Train set size: {len(train_df)} events, {len(train_df['trace_id'].unique())} traces")
        print(f"Test set size: {len(test_df)} events, {len(test_df['trace_id'].unique())} traces")
        print("--------------------")

        # Analyze distributions
        analyze_split_distributions(dataset_name)

    
    return results

if __name__ == "__main__":
    process_all_datasets()