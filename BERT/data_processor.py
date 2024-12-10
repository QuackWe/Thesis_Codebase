import pandas as pd
from sys import argv
import os

log = argv[1]
working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"
input_file = working_directory + "/Andrei_thesis_KRIF_"+log+"_vPaul_v2.csv"
# Create the directory if it doesn't exist
os.makedirs("datasets/"+log, exist_ok=True)


def preprocess_dataset(input_file, output_file):
    try:
        # Load the dataset with explicit encoding
        df = pd.read_csv(input_file, encoding='latin-1')  # Change encoding if needed

        # Combine topic and subtopic to form the Activity column
        df['Activity'] = df['topic'] + " - " + df['subtopic']

        # Keep only the required columns
        preprocessed_df = df[["TimestampContact", "CustomerId", "Activity", "outcome"]]

        # Rename columns
        preprocessed_df.rename(
            columns={
                "TimestampContact": "Timestamp",
                "CustomerId": "CaseID",
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


preprocess_dataset(input_file, output_file = "datasets/"+log+"/"+log+"_processed.csv")

# Run the preprocessing
preprocess_data_for_mam(
    input_file = "datasets/"+log+"/mortgages_processed.csv",
    output_prefix_file = "datasets/"+log+"/preprocessed_prefixes.csv",
    output_masked_file = None
)
