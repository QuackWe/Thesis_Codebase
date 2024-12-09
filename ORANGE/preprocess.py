import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# Adjust paths as needed
working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
file_path_mortgages = working_directory + "/Andrei_thesis_KRIF_mortgages_vPaul_v2.csv"
# If not used, you can remove file_path_applications
# file_path_applications = working_directory + "/Andrei_thesis_KRIF_application_vPaul_v2.csv"

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sys import argv

# Adjust these as needed
working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
raw_file_path = working_directory + "/Andrei_thesis_KRIF_mortgages_vPaul_v2.csv"
dataset = "mortgages"

# Directory to save preprocessed files
output_dir = f"dataset/{dataset}/"
os.makedirs(output_dir, exist_ok=True)

# Columns to keep after creating activity
columns_to_keep = [
    'CustomerId',
    'type_of_customer',
    'outcome',
    'activity',
    'TimestampContact'
]

# 1. Load raw data
df = pd.read_csv(raw_file_path, encoding='latin-1')

# If 'topic' and 'subtopic' exist, create activity
if 'topic' not in df.columns or 'subtopic' not in df.columns:
    raise ValueError("Data must contain 'topic' and 'subtopic' columns to create 'activity'.")

df['activity'] = df['topic'].astype(str) + '_' + df['subtopic'].astype(str)
# Now that activity is created, we can drop 'topic' and 'subtopic' if they're not needed
df.drop(['topic', 'subtopic'], axis=1, inplace=True)

# 2. Keep only the specified columns
missing_cols = [c for c in columns_to_keep if c not in df.columns]
if missing_cols:
    raise ValueError(f"These required columns are missing: {missing_cols}")
df = df[columns_to_keep].copy()

# 3. Drop rows with missing values
df.dropna(inplace=True)

# 4. Convert TimestampContact to datetime
df['TimestampContact'] = pd.to_datetime(df['TimestampContact'], errors='coerce')
df.dropna(subset=['TimestampContact'], inplace=True)

# Extract day_of_week and hour
df['day_of_week'] = df['TimestampContact'].dt.dayofweek
df['hour'] = df['TimestampContact'].dt.hour

# Drop TimestampContact if no longer needed
df.drop('TimestampContact', axis=1, inplace=True)

# Check for CustomerId and outcome
if 'CustomerId' not in df.columns or 'outcome' not in df.columns:
    raise ValueError("Data must contain 'CustomerId' and 'outcome' columns.")

# 5. Identify categorical columns (excluding outcome and CustomerId)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'outcome' in categorical_cols:
    categorical_cols.remove('outcome')
if 'CustomerId' in categorical_cols:
    categorical_cols.remove('CustomerId')

# One-hot encode categorical columns except outcome
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols)

# 6. Encode outcome if categorical
if df['outcome'].dtype == 'object':
    le = LabelEncoder()
    df['outcome'] = le.fit_transform(df['outcome'])

# 7. Group-based train/test split by CustomerId
case_ids = df['CustomerId'].unique()
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['CustomerId']))

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

y_train = df_train['outcome']
y_test = df_test['outcome']

# 8. Extract CustomerId
train_customer_ids = df_train['CustomerId']
test_customer_ids = df_test['CustomerId']

df_train.drop(['CustomerId', 'outcome'], axis=1, inplace=True)
df_test.drop(['CustomerId', 'outcome'], axis=1, inplace=True)

numeric_cols = df_train.columns.tolist()

# Convert floats to float32 for memory efficiency
for c in numeric_cols:
    if pd.api.types.is_float_dtype(df_train[c]):
        df_train[c] = df_train[c].astype('float32')
        df_test[c] = df_test[c].astype('float32')

# 9. Normalize numeric columns
scaler = MinMaxScaler()
scaler.fit(df_train[numeric_cols])

X_train_norm = pd.DataFrame(scaler.transform(df_train[numeric_cols]), columns=numeric_cols)
X_test_norm = pd.DataFrame(scaler.transform(df_test[numeric_cols]), columns=numeric_cols)

X_train_norm['outcome'] = y_train.reset_index(drop=True)
X_test_norm['outcome'] = y_test.reset_index(drop=True)

# 10. Save train_norm and test_norm
X_train_norm.to_csv(os.path.join(output_dir, f"{dataset}_train_norm.csv"), index=False)
X_test_norm.to_csv(os.path.join(output_dir, f"{dataset}_test_norm.csv"), index=False)

# 11. Create len_test<dataset>.csv
# Use df_test_full (original df rows for test) before drop
df_test_full = df.iloc[test_idx].copy()

# Sort by CustomerId and possibly by day_of_week/hour if needed
df_test_full.sort_values(by=['CustomerId', 'day_of_week', 'hour'], inplace=True)

len_data = []
for cid, group in df_test_full.groupby('CustomerId'):
    for i, _ in enumerate(group.index, start=1):
        len_data.append([cid, i])

len_df = pd.DataFrame(len_data, columns=["CaseID", "Len"])
len_df.to_csv(os.path.join(output_dir, f"len_test{dataset}.csv"), index=False)

print("Preprocessing complete. Generated files:")
print(f"- {dataset}_train_norm.csv")
print(f"- {dataset}_test_norm.csv")
print(f"- len_test{dataset}.csv")