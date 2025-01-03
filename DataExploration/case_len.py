import pandas as pd
import numpy as np
from sys import argv

# Input file
log = argv[1]
working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
input_file = f"{working_directory}/Andrei_thesis_KRIF_{log}_vPaul_v2.csv"

# Load the input data
df = pd.read_csv(input_file, encoding='latin-1')

# Combine Topic and Subtopic to create the Activity column
df['Activity'] = df['topic'].astype(str) + "_" + df['subtopic'].astype(str)

# Sort and group by CustomerId
df = df.sort_values(by=['CustomerId', 'TimestampContact'])
trace_lengths = df.groupby('CustomerId')['Activity'].apply(len)  # Compute length of each trace

# Compute statistics
max_length = trace_lengths.max()
min_length = trace_lengths.min()
mean_length = trace_lengths.mean()
std_dev_length = trace_lengths.std()

# Print the results
print("=== Trace Length Statistics ===")
print(f"Maximum Trace Length: {max_length}")
print(f"Minimum Trace Length: {min_length}")
print(f"Mean Trace Length: {mean_length:.2f}")
print(f"Standard Deviation of Trace Length: {std_dev_length:.2f}")
