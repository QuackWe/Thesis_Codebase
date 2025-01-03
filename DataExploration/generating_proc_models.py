import pandas as pd
import pm4py

# Load and format the DataFrame
df = pm4py.format_dataframe(
    pd.read_csv('normalized_timestamps.csv'),
    case_id='CustomerId',
    activity_key='Activity',
    timestamp_key='TimestampContact',
)

# Parameters
freq = 0.2  # Frequency threshold for filtering edges in the process model

# Separate process models for "Success" and "No Success"
outcome_success = df[df['outcome'] == "Success"]
outcome_no_success = df[df['outcome'] == "No Success"]
outcome_transit = df[df['outcome'] == "Transit"]

# Function to generate and save BPMN model
def generate_bpmn_model(df_subset, outcome_label, freq_threshold):
    if df_subset.empty:
        print(f"No data found for outcome: {outcome_label}")
        return

    # Discover BPMN model using the inductive miner
    bpmn_model = pm4py.discover_bpmn_inductive(df_subset, freq_threshold)

    # Save the BPMN model visualization to an image file
    output_image_path = f"bpmn_model_{outcome_label}_{freq_threshold}.png"
    pm4py.save_vis_bpmn(bpmn_model, output_image_path)
    print(f"BPMN model for '{outcome_label}' saved to {output_image_path}")

# Generate models
# print("Generating BPMN model for cases with 'No Success' outcome...")
# generate_bpmn_model(outcome_no_success, "NoSuccess", freq)
#
# print("Generating BPMN model for cases with 'Transit' outcome...")
# generate_bpmn_model(outcome_transit, "Transit", freq)

print("Generating BPMN model for cases with 'Success' outcome...")
generate_bpmn_model(outcome_success, "Success", 0.02)
generate_bpmn_model(outcome_success, "Success", 0.005)
generate_bpmn_model(outcome_success, "Success", 0.002)




