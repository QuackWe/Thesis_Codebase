import os
import pandas as pd
import torch
from sympy.stats.sampling.sample_numpy import numpy
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from sys import argv


log = argv[1]
dataset_path = 'datasets/'+log
working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
file_path_mortgages = working_directory + "/Andrei_thesis_KRIF_mortgages_vPaul_v2.csv"
file_path_applications = working_directory + "/Andrei_thesis_KRIF_application_vPaul_v2.csv"
# Create the directory if it doesn't exist
os.makedirs("datasets/"+log, exist_ok=True)


def load_mortgages(path):
    # Load data
    df_mortgages = pd.read_csv(path, encoding="ISO-8859-1")

    # Select necessary columns
    df_mortgages = df_mortgages[['CustomerId', 'topic', 'subtopic', 'TimestampContact', 'outcome']]

    # Rename CustomerId to CaseID and TimestampContact to Timestamp
    df_mortgages.rename(columns={
        'CustomerId': 'CaseID',
        'TimestampContact': 'Timestamp',
        'outcome': 'final_outcome',
    }, inplace=True)

    # Create ActivityName by concatenating topic and subtopic with a separator
    df_mortgages['Activity'] = df_mortgages['topic'] + " - " + df_mortgages['subtopic']

    # Drop the now redundant topic and subtopic columns
    df_mortgages.drop(columns=['topic', 'subtopic'], inplace=True)

    # Sorting by caseID and timestamp
    df_mortgages = df_mortgages.sort_values(by=['CaseID', 'Timestamp'])

    # Drop rows that include NaNs
    df_mortgages.dropna(inplace=True)

    # Grouping events by each case (process instance)
    mortgages_traces = df_mortgages.groupby('CaseID').agg(list)

    return df_mortgages, mortgages_traces


def calculate_time_features(df):
    # Convert the Timestamp to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', errors='coerce')

    # Ensure no invalid dates remain after coercion
    if df['Timestamp'].isna().any():
        raise ValueError("Some Timestamps could not be parsed. Check the data format.")

    # Initialize time columns
    df['next_event_time'] = 0
    df['remaining_time'] = 0

    # Iterate through each case to calculate next event time and remaining time
    for case_id, group in df.groupby('CaseID'):
        timestamps = group['Timestamp'].values
        num_events = len(timestamps)

        # Calculate next event time and remaining time for each event in the trace
        for i in range(num_events):
            if i < num_events - 1:
                df.loc[group.index[i], 'next_event_time'] = (timestamps[i + 1] - timestamps[i]).astype(
                    'timedelta64[s]').astype(np.int32)
            df.loc[group.index[i], 'remaining_time'] = (timestamps[-1] - timestamps[i]).astype('timedelta64[s]').astype(
                np.int32)

    return df
    # # Drop rows where Timestamp is NaT
    # df_mortgages.dropna(subset=['Timestamp'], inplace=True)
    #

def generate_labels(df):
    # Initialize labels
    df['next_activity'] = df['Activity'].shift(-1)

    # For the last event in each case, set next_activity as NaN or a special label
    for case_id, group in df.groupby('CaseID'):
        df.loc[group.index[-1], 'next_activity'] = 'END'  # Marking last event's next activity as 'END'

    return df

df_mortgages, mortgages_traces = load_mortgages(file_path_mortgages)
df_mortgages = calculate_time_features(df_mortgages)

# One-hot encode the activity column
df_mortgages_onehot = pd.get_dummies(df_mortgages, columns=['Activity'])

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize the next_event_time and remaining_time columns
df_mortgages[['next_event_time', 'remaining_time']] = scaler.fit_transform(df_mortgages[['next_event_time', 'remaining_time']])

df_mortgages = generate_labels(df_mortgages)

# Features (activity sequences in one-hot encoded format)
X = df_mortgages_onehot.filter(like='Activity_').values
torch.save(torch.tensor(X, dtype=torch.float32), os.path.join(dataset_path, "X.pt"))

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the next_activity column
df_mortgages['next_activity_encoded'] = label_encoder.fit_transform(df_mortgages['next_activity'])

# Fit and transform the final_outcome column
df_mortgages['final_outcome_encoded'] = label_encoder.fit_transform(df_mortgages['final_outcome'])


# Now you can convert the encoded labels into tensors
y_activity = df_mortgages['next_activity_encoded'].values
torch.save(torch.tensor(y_activity, dtype=torch.long), os.path.join(dataset_path, "y_activity.pt"))

# Now you can convert the encoded labels into tensors
y_outcome = df_mortgages['final_outcome_encoded'].values
torch.save(torch.tensor(y_outcome, dtype=torch.long), os.path.join(dataset_path, "y_outcome.pt"))

# Next event time (target for regression)
y_next_time = df_mortgages['next_event_time'].values
torch.save(torch.tensor(y_next_time, dtype=torch.float32), os.path.join(dataset_path, "y_next_time.pt"))

# Remaining time (target for regression)
y_remaining_time = df_mortgages['remaining_time'].values
torch.save(torch.tensor(y_remaining_time, dtype=torch.float32), os.path.join(dataset_path, "y_remaining_time.pt"))

# # Display the processed
# with pd.option_context('display.max_columns', 85):
#     print(df_mortgages.head())