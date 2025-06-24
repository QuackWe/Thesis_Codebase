import pandas as pd
import numpy as np
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True, help='Base output directory')
parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
args = parser.parse_args()
dataset = args.log
output_dir = args.output_dir

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


def preprocess_df(df):
    """Helper function to process a dataframe"""
    # Create activity column
    if 'topic' not in df.columns or 'subtopic' not in df.columns:
        raise ValueError("Data must contain 'topic' and 'subtopic' columns")
    
    df['activity'] = df['topic'].astype(str) + '_' + df['subtopic'].astype(str)
    df.drop(['topic', 'subtopic'], axis=1, inplace=True)
    
    # Keep only specified columns
    columns_to_keep = ['trace_id', 'type_of_customer', 'outcome', 'activity', 'TimestampContact']
    missing_cols = [c for c in columns_to_keep if c not in df.columns]
    if missing_cols:
        raise ValueError(f"These required columns are missing: {missing_cols}")
    df = df[columns_to_keep].copy()
    
    # Handle timestamps
    df['TimestampContact'] = pd.to_datetime(df['TimestampContact'], errors='coerce')
    df.dropna(subset=['TimestampContact'], inplace=True)
    df['day_of_week'] = df['TimestampContact'].dt.dayofweek
    df['hour'] = df['TimestampContact'].dt.hour
    df.drop('TimestampContact', axis=1, inplace=True)
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ['outcome', 'trace_id']]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols)
    
    return df

# Paths setup
raw_file_path_trainval = f"../../datasets/{dataset}/{dataset}_train-val.csv"
raw_file_path_test = f"../../datasets/{dataset}/{dataset}_test.csv"

# Columns to keep after creating activity
columns_to_keep = [
    'trace_id',
    'type_of_customer',
    'outcome',
    'activity',
    'TimestampContact'
]

# Load and preprocess train-val and test separately
df_trainval = pd.read_csv(raw_file_path_trainval, encoding='latin-1')
df_test = pd.read_csv(raw_file_path_test, encoding='latin-1')

# Apply trace identifier
# df_trainval = get_trace_identifier(df_trainval, dataset)
# df_test = get_trace_identifier(df_test, dataset)

# After loading the datasets and before preprocessing, add this code:
df_trainval['outcome'] = df_trainval['outcome'].fillna('Transit')
df_test['outcome'] = df_test['outcome'].fillna('Transit')

# Preprocess both datasets
df_trainval = preprocess_df(df_trainval)
df_test = preprocess_df(df_test)

# Store trace_ids and outcomes
trainval_ids = df_trainval['trace_id']
test_ids = df_test['trace_id']
y_trainval = df_trainval['outcome']
y_test = df_test['outcome']

# Create and fit label encoder
le = LabelEncoder()
y_trainval = pd.Series(le.fit_transform(y_trainval))
y_test = pd.Series(le.transform(y_test))

# Save the label mapping for reference
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", label_mapping)

# Drop trace_id and outcome
df_trainval.drop(['trace_id', 'outcome'], axis=1, inplace=True)
df_test.drop(['trace_id', 'outcome'], axis=1, inplace=True)

# Ensure both datasets have the same columns
all_columns = list(set(df_trainval.columns) | set(df_test.columns))
for col in all_columns:
    if col not in df_trainval.columns:
        df_trainval[col] = 0
    if col not in df_test.columns:
        df_test[col] = 0

# Convert floats to float32
numeric_cols = df_trainval.columns.tolist()
for c in numeric_cols:
    if pd.api.types.is_float_dtype(df_trainval[c]):
        df_trainval[c] = df_trainval[c].astype('float32')
        df_test[c] = df_test[c].astype('float32')

# Fit scaler on train-val and transform both datasets
scaler = MinMaxScaler()
scaler.fit(df_trainval[numeric_cols])

X_trainval_norm = pd.DataFrame(scaler.transform(df_trainval[numeric_cols]), columns=numeric_cols)
X_test_norm = pd.DataFrame(scaler.transform(df_test[numeric_cols]), columns=numeric_cols)

# Add outcomes back
X_trainval_norm['outcome'] = y_trainval.reset_index(drop=True)
X_test_norm['outcome'] = y_test.reset_index(drop=True)

# Save normalized datasets
X_trainval_norm.to_csv(os.path.join(output_dir, f"{dataset}_train_norm.csv"), index=False)
X_test_norm.to_csv(os.path.join(output_dir, f"{dataset}_test_norm.csv"), index=False)

# Create len_test file for test data
test_df = df_test.copy()
test_df['trace_id'] = test_ids.reset_index(drop=True)
test_df.sort_values(by=['trace_id', 'day_of_week', 'hour'], inplace=True)

len_data = []
for cid, group in test_df.groupby('trace_id'):
    for i, _ in enumerate(group.index, start=1):
        len_data.append([cid, i])

len_df = pd.DataFrame(len_data, columns=["CaseID", "Len"])
len_df.to_csv(os.path.join(output_dir, f"len_test{dataset}.csv"), index=False)

print("Preprocessing complete. Generated files:")
print(f"- {dataset}_train_norm.csv")
print(f"- {dataset}_test_norm.csv")
print(f"- len_test{dataset}.csv")