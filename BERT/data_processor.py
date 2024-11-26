import pandas as pd


working_directory = "K:/Klanten/De Volksbank/Thesis Andrei"

file_path_mortgages = working_directory + "/Andrei_thesis_KRIF_mortgages_vPaul_v2.csv"
file_path_applications = working_directory + "/Andrei_thesis_KRIF_application_vPaul_v2.csv"


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

# Example usage
input_file = file_path_mortgages # Replace with your input file path
output_file = "mortgages_processed.csv"  # Replace with your desired output file path
preprocess_dataset(input_file, output_file)

