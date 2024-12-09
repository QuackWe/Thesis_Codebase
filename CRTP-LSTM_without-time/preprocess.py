import pandas as pd
import re

working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
file_path_mortgages = working_directory + "/Andrei_thesis_KRIF_mortgages_vPaul_v2.csv"
file_path_applications = working_directory + "/Andrei_thesis_KRIF_application_vPaul_v2.csv"

# Load your raw dataset
raw_data = pd.read_csv(file_path_mortgages, encoding='latin-1')

# Rename CustomerId to CaseID
raw_data = raw_data.rename(columns={'CustomerId': 'CaseID'})

# Handle inconsistent timestamp formats by adding missing seconds if necessary
raw_data['TimestampContact'] = raw_data['TimestampContact'].apply(
    lambda x: re.sub(r'(T\d{2}:\d{2})$', r'\1:00', x)
)
raw_data['TimestampContact'] = pd.to_datetime(raw_data['TimestampContact'], errors='coerce')

# Drop rows with NaN values resulting from invalid timestamps
raw_data = raw_data.dropna(subset=['TimestampContact'])

# Sort by CaseID and TimestampContact
raw_data = raw_data.sort_values(['CaseID', 'TimestampContact'])

# Create the Activity column by combining topic and subtopic
raw_data['Activity'] = raw_data['topic'] + ' - ' + raw_data['subtopic']

# Compute the time differences within each CaseID
def compute_time_diffs(group):
    group = group.copy()
    group['time_diff'] = group['TimestampContact'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().astype(int)
    group['time_since_start'] = (group['TimestampContact'] - group['TimestampContact'].min()).dt.total_seconds().astype(int)
    group['remtime_std'] = group['time_since_start'].std()  # Calculate the standard deviation as a representation for remtime_std
    return group

# Apply the function and reset the index to avoid ambiguity
raw_data = raw_data.groupby('CaseID').apply(compute_time_diffs).reset_index(drop=True)

# Drop any remaining NaN values
raw_data = raw_data.dropna()

# Aggregate the data per CaseID
aggregated_data = raw_data.groupby('CaseID').agg({
    'Activity': lambda x: ', '.join(x),
    'time_diff': lambda x: ', '.join(map(str, x.astype(int))),
    'time_since_start': lambda x: ', '.join(map(str, x.astype(int))),
    'topic': lambda x: ', '.join(x),
    'subtopic': lambda x: ', '.join(x),
    'remtime_std': 'mean'  # Aggregate the standard deviation by taking the mean across the case
}).reset_index()

# Rename columns accordingly
aggregated_data.rename(columns={
    'Activity': 'trace',
    'time_diff': 'time_column1',
    'time_since_start': 'time_column2',
    'topic': 'cat_column1',
    'subtopic': 'cat_column2',
}, inplace=True)

# For numerical columns, use lists of integers for consistency with model expectations
aggregated_data['num_column1'] = aggregated_data['remtime_std'].apply(lambda x: [int(x)] if not pd.isna(x) else [0])  # Replace NaN with a default value like 0
aggregated_data['num_column2'] = aggregated_data['time_column2'].apply(lambda x: [int(i) for i in x.split(', ')])

# Drop any rows with NaNs after aggregation (to ensure clean data)
aggregated_data = aggregated_data.dropna()

# Select only the necessary columns for the final dataset
final_columns = ['CaseID', 'trace', 'time_column1', 'time_column2', 'cat_column1', 'cat_column2', 'num_column1', 'num_column2']
final_data = aggregated_data[final_columns]

# Save the final dataset to a CSV file
final_data.to_csv('data/data.csv', index=False)

# Create train, validation, and test index files
case_ids = final_data['CaseID'].unique()
train_ids = case_ids[:int(0.7 * len(case_ids))]
valid_ids = case_ids[int(0.7 * len(case_ids)):int(0.85 * len(case_ids))]
test_ids = case_ids[int(0.85 * len(case_ids)):]

pd.DataFrame(train_ids, columns=['CaseID']).to_csv('data/train_index.csv', index=False)
pd.DataFrame(valid_ids, columns=['CaseID']).to_csv('data/valid_index.csv', index=False)
pd.DataFrame(test_ids, columns=['CaseID']).to_csv('data/test_index.csv', index=False)
