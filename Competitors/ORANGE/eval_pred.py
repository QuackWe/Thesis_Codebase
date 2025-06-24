import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import pickle as pk
import os
import argparse

def evaluate_model(dataset_name, model, X_test, y_test, pd_len, output_dir):
    """Evaluate model and save predictions in standardized format"""
    max_len = pd_len['Len'].max()
    
    # Store all predictions in standardized format
    all_predictions = {
        'prefix_length': [],
        'outcome_true': [],
        'outcome_pred': [],
        'outcome_probs': []
    }
    
    # Process predictions by prefix length
    index_len = 1
    i = 0
    while i < max_len:
        # Get samples for current prefix length
        j = 0
        image_test = []
        target_test = []
        while j < len(pd_len):
            val = pd_len.iloc[j]['Len']
            if val == index_len:
                image_test.append(X_test[j])
                target_test.append(y_test[j])
            j = j + 1
        
        if len(image_test) > 0:
            conv_test = np.asarray(image_test)
            conv_y_test = np.asarray(target_test)
            
            # Get predictions
            pred_probs = model.predict(conv_test)
            pred_classes = np.argmax(pred_probs, axis=1)
            
            # Store predictions
            all_predictions['prefix_length'].extend([index_len] * len(target_test))
            all_predictions['outcome_true'].extend(target_test)
            all_predictions['outcome_pred'].extend(pred_classes)
            all_predictions['outcome_probs'].extend(pred_probs)
        
        index_len += 1
        i += 1
    
    # Convert to DataFrame
    results_df = pd.DataFrame({
        'prefix_length': all_predictions['prefix_length'],
        'outcome_true': all_predictions['outcome_true'],
        'outcome_pred': all_predictions['outcome_pred']
    })
    
    # Add dummy columns for activity prediction (since this is outcome-only model)
    results_df['activity_true'] = -1  # Use -1 to indicate N/A
    results_df['activity_pred'] = -1
    
    # Add probability columns for outcome classes
    outcome_probs = np.array(all_predictions['outcome_probs'])
    for i in range(outcome_probs.shape[1]):
        results_df[f'outcome_prob_{i}'] = outcome_probs[:, i]
    
    # Save predictions to CSV
    results_df.to_csv(f"{output_dir}/predictions.csv", index=False)
    print(f"Predictions saved to {output_dir}/predictions.csv")
    
    return results_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='Base output directory')
    parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
    args = parser.parse_args()
    dataset_name = args.log
    output_dir = args.output_dir
    
    # Load model and data
    model = load_model(f"{output_dir}/{dataset_name}.h5")
    pd_len = pd.read_csv(f"{output_dir}/len_test{dataset_name}.csv")
    
    # Load test data
    with open(output_dir+"/"+output_dir.split('/')[2]+"_test.pickle","rb") as pickle_test:
        X_test = pk.load(pickle_test)
    image_all = np.asarray(X_test)
    image_size = image_all.shape[1]
    image_all = np.reshape(image_all, [-1, image_size, image_size, 1])
    
    y_test = pd.read_csv(f"{output_dir}/{dataset_name}_test_norm.csv")
    y_test = y_test[y_test.columns[-1]]
    y_test = y_test.astype(int)
    
    # Evaluate model and save predictions
    results_df = evaluate_model(dataset_name, model, image_all, y_test, pd_len, output_dir)
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()