import pandas as pd
import os

# log = argv[1]
log = 'mortgages'
working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
input_file = working_directory + "/Andrei_thesis_KRIF_"+log+"_vPaul_v2.csv"
# Create the directory if it doesn't exist
# os.makedirs("datasets/"+log, exist_ok=True)


# Load the CSV file
df = pd.read_csv(input_file, encoding='latin-1')

# Get an overview of missing values
missing_summary = df.isnull().sum()

# Display columns with missing values
print(missing_summary[missing_summary > 0])

import missingno as msno

# Visualize missing data
msno.matrix(df)

# # Calculate the count of each unique value in the 'outcome' column
# outcome_counts = df['outcome'].value_counts()
#
# # Calculate the percentage of each unique value
# outcome_percentages = df['outcome'].value_counts(normalize=True) * 100
#
# # Combine the count and percentage into a single DataFrame
# outcome_summary = pd.DataFrame({
#     'Count': outcome_counts,
#     'Percentage': outcome_percentages
# })
#
# # Display the result
# print(outcome_summary)
#
# # Calculate the count of each unique value in the 'type_of_customer' column
# customer_type_counts = df['type_of_customer'].value_counts()
#
# # Calculate the percentage of each unique value
# customer_type_percentages = df['type_of_customer'].value_counts(normalize=True) * 100
#
# # Combine the count and percentage into a single DataFrame
# customer_type_summary = pd.DataFrame({
#     'Count': customer_type_counts,
#     'Percentage': customer_type_percentages
# })
#
# # Display the result
# print(customer_type_summary)
