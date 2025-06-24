import pandas as pd
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True, help='Base output directory')
parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
args = parser.parse_args()
log = args.log
output_dir = args.output_dir
train_val = f"../../datasets/{log}/{log}_train-val.csv"
test = f"../../datasets/{log}/{log}_test.csv"

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
    elif dataset_type == 'mortgages':
        df['trace_id'] = df['CustomerId'].astype(str)
    elif dataset_type == 'bpic2017':
        df['trace_id'] = df['case:concept:name'].astype(str)
        df['topic'] = df['EventOrigin']
        df['subtopic'] = df['Action']
        df['type_of_customer'] = df['case:ApplicationType']
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
    else:
        print('Dataset not recognized.')
    return df

def preprocess_dataset(input_file, output_file):
    try:
        # # Load your raw dataset
        # if log == 'application':
        #     # Load data
        #     df = pd.read_csv(input_file, encoding='latin-1', sep='|')
        #     df = get_trace_identifier(df, 'application')
        # elif log == 'mortgages':
        #     df = pd.read_csv(input_file, encoding='latin-1')
        #     df = get_trace_identifier(df, 'mortgages')
        # elif log == 'bpic2017':
        #     # Load and convert to DataFrame
        #     df = pd.DataFrame([
        #         {**event, **{'case:' + k: v for k, v in trace.attributes.items()}}
        #         for trace in xes_importer.apply(f'datasets/{log}/BPI_Challenge_2017.xes')
        #         for event in trace
        #     ])
        #     df = get_trace_identifier(df, 'bpic2017')

        # # Combine topic and subtopic to form the Activity column
        # df['Activity'] = df['topic'] + " - " + df['subtopic']
        df = pd.read_csv(input_file, encoding='latin-1')

        # Keep only the required columns
        preprocessed_df = df[["TimestampContact", "trace_id", "Activity", "outcome"]]

        # Rename columns
        preprocessed_df.rename(
            columns={
                "TimestampContact": "Timestamp",
                "trace_id": "CaseID",
                "outcome": "FinalOutcome",
            },
            inplace=True,
        )

        # Save the preprocessed dataset to a new CSV file
        preprocessed_df.to_csv(output_file, index=False)
        print(f"Preprocessed dataset saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


def preprocess_data_for_mam(input_file, output_prefix_file, output_masked_file):
    # Load the processed dataset
    df = pd.read_csv(input_file)

    # Group data by CaseID to create traces
    grouped = df.groupby("CaseID")["Activity"].apply(list).reset_index()
    grouped.rename(columns={"Activity": "Trace"}, inplace=True)

    # Generate prefixes and corresponding masked versions
    prefixes = []
    masked_activities = []

    for _, row in grouped.iterrows():
        trace = row["Trace"]
        for i in range(1, len(trace)):
            prefix = trace[:i]  # Get prefix
            masked = trace[i]  # Masked activity (target)
            prefixes.append(prefix)
            masked_activities.append(masked)

    # Create a DataFrame for prefixes and masked activities
    prefix_df = pd.DataFrame({"Prefix": prefixes, "MaskedActivity": masked_activities})

    # Save the data
    prefix_df.to_csv(output_prefix_file, index=False)
    print(f"Preprocessed prefixes and masked activities saved to {output_prefix_file}")


preprocess_dataset(train_val, output_file = output_dir+"/"+log+"_processed_train-val.csv")
preprocess_dataset(test, output_file = output_dir+"/"+log+"_processed_test.csv")

# Run the preprocessing
preprocess_data_for_mam(
    input_file = output_dir+"/"+log+"_processed_train-val.csv",
    output_prefix_file = output_dir+"/preprocessed_prefixes_train-val.csv",
    output_masked_file = None
)
preprocess_data_for_mam(
    input_file = output_dir+"/"+log+"_processed_test.csv",
    output_prefix_file = output_dir+"/preprocessed_prefixes_test.csv",
    output_masked_file = None
)