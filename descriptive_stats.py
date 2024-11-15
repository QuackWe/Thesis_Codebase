import pandas as pd
from vars import file_path_mortgages, file_path_applications

for file_path in [file_path_mortgages, file_path_applications]:
    print(file_path)
    df = pd.read_csv(file_path, encoding="ISO-8859-1")  # or try "latin1" if this doesn't work

    # Basic overview
    print("Basic Information:")
    print(df.info())  # Get info about data types and non-null counts

    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe(include='all'))  # Descriptive statistics for numerical and categorical data

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Exclude non-numeric columns

    # Correlation matrix for numerical features
    print("\nCorrelation Matrix:")
    print(numeric_df.corr())