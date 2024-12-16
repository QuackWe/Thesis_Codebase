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


if __name__ == "__main__":
    # Command-line arguments
    log = argv[1]
    working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
    input_file = f"{working_directory}/Andrei_thesis_KRIF_{log}_vPaul_v2.csv"

    # Create the output directory
    os.makedirs(f"datasets/{log}", exist_ok=True)

    # Load the input data
    df = pd.read_csv(input_file, encoding='latin-1')

    # Normalize timestamps
    df = normalize_timestamps(df, 'TimestampContact')

    # Combine Topic and Subtopic to create the Activity column
    df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)
    df = df.sort_values(by=['CustomerId', 'TimestampContact'])

    # Group by CustomerId to create traces
    traces = df.groupby('CustomerId')['Activity'].apply(list).reset_index()
    outcomes = df.groupby('CustomerId')['outcome'].first().reset_index()
    types = df.groupby('CustomerId')['type_of_customer'].first().reset_index()

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
    outcomes['OutcomeID'] = outcome_encoder.fit_transform(outcomes['outcome'])

    # Convert traces to padded sequences
    max_seq_length = max(len(trace) for trace in traces['ActivityIDs'])
    print(f"Max sequence length determined: {max_seq_length}")

    padded_traces = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(trace[:max_seq_length]) for trace in traces['ActivityIDs']],
        batch_first=True,
        padding_value=0
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
        'AttentionMask': attention_masks.tolist(),
        'Outcome': outcomes['OutcomeID'],
        'CustomerType': types['TypeID']
    })

    # print(dataset.head())
    # print(len(dataset['Trace'][0]))
    # print(max(len(trace) for trace in dataset['Trace']))

    # Create PyTorch Dataset and save it
    torch_dataset = TraceDataset(dataset)
    torch.save(torch_dataset, f"datasets/{log}/pytorch_dataset.pt")
    print(f"Processed dataset saved to datasets/{log}/pytorch_dataset.pt")
