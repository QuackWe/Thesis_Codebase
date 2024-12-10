import pandas as pd
from sys import argv
from sklearn.model_selection import train_test_split
log = argv[1]


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


# Define input and output files
input_file = "datasets/"+log+"/mortgages_processed.csv"  # Processed dataset
output_prefix_file = "datasets/"+log+"/preprocessed_prefixes.csv"  # Output for prefixes and masked activities

# Run the preprocessing
preprocess_data_for_mam(input_file, output_prefix_file, None)
