import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertModel,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.modeling_bert import BertPooler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse
import time


def print_outcome_class_distribution(prefix_lengths, labels):
    results_df = pd.DataFrame({
        'prefix_length': prefix_lengths,
        'outcome_true': labels
    })
    
    # Group by prefix length and count unique outcomes
    outcome_dist = results_df.groupby('prefix_length')['outcome_true'].agg([
        ('num_classes', 'nunique'),
        ('total_samples', 'count')
    ]).reset_index()

    # Filter out prefix lengths with <1 sample
    outcome_dist = outcome_dist[outcome_dist['total_samples'] > 0]
    
    print("\nOutcome Class Distribution per Prefix Length:")
    print("Length  Classes  Samples")
    for _, row in outcome_dist.iterrows():
        print(f"{int(row['prefix_length']):<6} {row['num_classes']:<7} {row['total_samples']}")


# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True, help='Base output directory')
parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
args = parser.parse_args()
log = args.log
output_dir = args.output_dir


# Load the tokenizer and pre-trained MAM model
tokenizer = BertTokenizer.from_pretrained(output_dir+'/mam_pretrained_model')
mam_model = BertForMaskedLM.from_pretrained(output_dir+'/mam_pretrained_model')

# Load the configuration from the MAM model
config = mam_model.config

# Load and preprocess the data
event_log_file_train_val = output_dir+'/'+log+'_processed_train-val.csv'
event_log_file_test = output_dir+'/'+log+'_processed_test.csv'
train_val_df = pd.read_csv(event_log_file_train_val, parse_dates=['Timestamp'])
train_val_df = train_val_df.sort_values(by=['CaseID', 'Timestamp'])
test_df = pd.read_csv(event_log_file_test, parse_dates=['Timestamp'])
test_df = test_df.sort_values(by=['CaseID', 'Timestamp'])

# Add these parameters after the initial imports
MAX_SAMPLES = 20  # Small number for testing
USE_SAMPLE_LIMIT = False  # Flag to toggle sample limiting

# Modify the data loading section after loading the CSVs
if USE_SAMPLE_LIMIT:
    train_val_df = train_val_df.head(MAX_SAMPLES)
    # test_df = test_df.head(MAX_SAMPLES // 2)  # Using fewer test samples
    print(f"Limited dataset to {len(train_val_df)} train/val samples and {len(test_df)} test samples")

# Group by CaseID to form traces
grouped_train_val = train_val_df.groupby('CaseID')
grouped_test = test_df.groupby('CaseID')

train_val_data = []
test_data = []
# Print initial data sizes
print(f"Initial train_val_df size: {len(train_val_df)}")
print(f"Initial test_df size: {len(test_df)}")
print(f"Number of unique CaseIDs in train_val: {train_val_df['CaseID'].nunique()}")
print(f"Number of unique CaseIDs in test: {test_df['CaseID'].nunique()}")

for case_id, group in grouped_train_val:
    activities_train_val = group['Activity'].tolist()
    final_outcome_train_val = group['FinalOutcome'].iloc[0]  # Assuming FinalOutcome is the same for all events in a case

    # Create prefixes
    for i in range(1, len(activities_train_val) + 1):
        prefix = activities_train_val[:i]
        train_val_data.append({
            'CaseID': case_id,
            'Prefix': prefix,
            'PrefixLength': len(prefix),
            'FinalOutcome': final_outcome_train_val
        })

for case_id, group in grouped_test:
    activities_test = group['Activity'].tolist()
    final_outcome_test = group['FinalOutcome'].iloc[0]


    for i in range(1, len(activities_test) + 1):
        prefix = activities_test[:i]
        test_data.append({
            'CaseID': case_id,
            'Prefix': prefix,
            'PrefixLength': len(prefix),
            'FinalOutcome': final_outcome_test
        })

# Create DataFrame
dataset_train_val_df = pd.DataFrame(train_val_data)
dataset_test_df = pd.DataFrame(test_data)

print("\nPrefix Statistics:")
print(f"Train-val prefixes created: {len(dataset_train_val_df)}")
print(f"Test prefixes created: {len(dataset_test_df)}")
print(f"Unique CaseIDs in train-val prefixes: {dataset_train_val_df['CaseID'].nunique()}")
print(f"Unique CaseIDs in test prefixes: {dataset_test_df['CaseID'].nunique()}")

# Get the list of unique outcomes from the 'FinalOutcome' column
unique_outcomes = dataset_train_val_df['FinalOutcome'].unique().tolist()
num_labels = len(unique_outcomes)
config.num_labels = num_labels

# # Create label mappings
label_map = {outcome: idx for idx, outcome in enumerate(unique_outcomes)}
id_to_label = {idx: outcome for outcome, idx in label_map.items()}

# Save label mappings for future use
with open(output_dir+'/outcome_label_map.json', 'w') as f:
    json.dump(label_map, f)

# Initialize a new BertModel without the pooling layer
bert_model = BertModel(config, add_pooling_layer=False)

# Manually add and initialize the pooler layer
bert_model.pooler = BertPooler(config)
bert_model.pooler.apply(bert_model._init_weights)

# Load the pre-trained weights from the MAM model into bert_model
bert_model.load_state_dict(mam_model.bert.state_dict(), strict=False)

# Initialize a new BertForSequenceClassification model
model = BertForSequenceClassification(config)

# Replace the bert encoder in model with our bert_model
model.bert = bert_model

# Move the model to the device
model.to(device)

# Split train_val_df into train and validation sets
train_df, val_df = train_test_split(
    dataset_train_val_df,
    test_size=0.1,
    random_state=42,
    stratify=dataset_train_val_df['FinalOutcome'],
)
test_df = dataset_test_df

# Save the datasets to CSV
train_df.to_csv(output_dir+'/outcome_train.csv', index=False)
val_df.to_csv(output_dir+'/outcome_val.csv', index=False)
test_df.to_csv(output_dir+'/outcome_test.csv', index=False)

print("Data splitting completed:")
print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")


# Define the OutcomePredictionDataset class
class OutcomePredictionDataset(Dataset):
    def __init__(self, data_file, tokenizer, label_map, max_len=128):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert string representation to list
        prefix = eval(self.data.iloc[idx]['Prefix'])
        prefix_length = len(prefix)  # Store the original prefix length
        final_outcome = self.data.iloc[idx]['FinalOutcome']
    
        # Convert the prefix into a string
        input_text = ' [SEP] '.join(prefix)

        # Tokenize the input
        encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            padding='max_length',
        )

        input_ids = encoding['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze()

        # Get the label ID
        label_id = self.label_map[final_outcome]

        # Generate position_ids
        position_ids = torch.arange(self.max_len, dtype=torch.long)  # Shape: [max_len]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': torch.tensor(label_id, dtype=torch.long),
            'original_prefix_length': torch.tensor(prefix_length, dtype=torch.long)
        }


# Create datasets and DataLoaders
train_dataset = OutcomePredictionDataset(output_dir+'/outcome_train.csv', tokenizer, label_map)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = OutcomePredictionDataset(output_dir+'/outcome_val.csv', tokenizer, label_map)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

test_dataset = OutcomePredictionDataset(output_dir+'/outcome_test.csv', tokenizer, label_map)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Debugging code
for i in range(5):  # Check 5 random examples
    sample = test_dataset[i]
    orig_len = sample['original_prefix_length'].item()
    token_len = sample['attention_mask'].sum().item()  # Count non-padding tokens
    print(f"Example {i}: Original prefix length = {orig_len}, Tokenized length = {token_len}")

# Add this before generating predictions DEBUG
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of batches in test loader: {len(test_loader)}")

# Set up the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10 #10
num_training_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

print("\nStarting training phase...")
training_start_time = time.time()

# Modify training loop:
# Fine-tuning loop
total_train_time = 0
total_val_time = 0
epoch_train_start = time.time()

# Fine-tuning loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]"
    )
    epoch_start_time = time.time()
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = batch['position_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix(
            {'loss': total_train_loss / (progress_bar.n + 1)}
        )

    epoch_train_time = time.time() - epoch_train_start
    total_train_time += epoch_train_time
    avg_train_loss = total_train_loss / len(train_loader)
    print(
        f"Epoch {epoch + 1} Training completed in {epoch_train_time:.2f}s. Average Loss: {avg_train_loss:.4f}"
    )

    # Validation phase
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_examples = 0
    epoch_val_start = time.time()

    with torch.no_grad():
        progress_bar = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"
        )
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_ids = batch['position_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            total_val_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    epoch_val_time = time.time() - epoch_val_start
    total_val_time += epoch_val_time
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_correct / total_examples
    print(
        f"Epoch {epoch + 1} Validation completed in {epoch_val_time:.2f}s. "
        f"Average Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
    )
    epoch_total_time = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} total time: {epoch_total_time:.2f}s")
    print("----------------------------------------")

training_time = time.time() - training_start_time
print(f"\nTotal training time: {training_time:.2f}s")
print(f"Average training time per epoch: {total_train_time/num_epochs:.2f}s")
print(f"Average validation time per epoch: {total_val_time/num_epochs:.2f}s")



def standardize_predictions(predictions_dict, trace_lengths, is_activity_predictor=True):
    """
    Standardize predictions format across all models
    """
    results_df = pd.DataFrame({
        'prefix_length': predictions_dict['prefix_length'],
        'case_id': predictions_dict.get('case_id', [-1] * len(predictions_dict['prefix_length'])),
    })
    
    # Add activity predictions (if available)
    if is_activity_predictor:
        results_df['activity_true'] = predictions_dict['activity_true']
        results_df['activity_pred'] = predictions_dict['activity_pred']
        # Add probability columns for activities
        activity_probs = np.array(predictions_dict['activity_probs'])
        for i in range(activity_probs.shape[1]):
            results_df[f'activity_prob_{i}'] = activity_probs[:, i]
        # Add dummy outcome columns
        results_df['outcome_true'] = -1
        results_df['outcome_pred'] = -1
        results_df['outcome_prob_0'] = np.nan
        results_df['outcome_prob_1'] = np.nan
    else:
        # Add dummy activity columns
        results_df['activity_true'] = -1
        results_df['activity_pred'] = -1
        results_df['activity_prob_0'] = np.nan
        # Add outcome predictions
        results_df['outcome_true'] = predictions_dict['outcome_true']
        results_df['outcome_pred'] = predictions_dict['outcome_pred']
        # Add probability columns for outcomes
        outcome_probs = np.array(predictions_dict['outcome_probs'])
        for i in range(outcome_probs.shape[1]):
            results_df[f'outcome_prob_{i}'] = outcome_probs[:, i]
    
    return results_df


# Evaluation on the test set
# Collect all predictions in standardized format
predictions = {
    'prefix_length': [],
    'case_id': [],
    'outcome_true': [],
    'outcome_pred': [],
    'outcome_probs': []
}

print("\nGenerating predictions...")
model.eval()
eval_start_time = time.time()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Generating predictions"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = batch['position_ids'].to(device)
        labels = batch['labels'].to(device)
        prefix_lengths = batch['original_prefix_length']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predictions['prefix_length'].extend(prefix_lengths.cpu().numpy())
        predictions['outcome_true'].extend(labels.cpu().numpy())
        predictions['outcome_pred'].extend(torch.argmax(logits, dim=1).cpu().numpy())
        predictions['outcome_probs'].extend(probs.cpu().numpy())

eval_time = time.time() - eval_start_time
print(f"\nTotal evaluation time: {eval_time:.2f}s")

# Save timing metrics to file
timing_metrics = {
    'total_training_time': training_time,
    'avg_training_time_per_epoch': total_train_time/num_epochs,
    'avg_validation_time_per_epoch': total_val_time/num_epochs,
    'total_evaluation_time': eval_time
}

with open(f"{output_dir}/timing_metrics_outcome.json", 'w') as f:
    json.dump(timing_metrics, f, indent=4)
print(f"Timing metrics saved to {output_dir}/timing_metrics_outcome.json")

print("\nPrediction Statistics:")
print(f"Number of predictions: {len(predictions['prefix_length'])}")
print(f"Number of unique prefix lengths: {len(set(predictions['prefix_length']))}")

# Load existing predictions if available
pred_file = f"{output_dir}/predictions.csv"

# Add this after collecting predictions DEBUG
print(f"Number of collected predictions: {len(predictions['prefix_length'])}")
if os.path.exists(pred_file):
    print(f"Number of rows in existing predictions file: {len(pd.read_csv(pred_file))}")

if os.path.exists(pred_file):
    existing_preds = pd.read_csv(pred_file)
    print(f"\nExisting predictions file statistics:")
    print(f"Number of rows: {len(existing_preds)}")
    print(f"Number of unique prefix lengths: {existing_preds['prefix_length'].nunique()}")
    # Verify alignment
    if len(existing_preds) != len(predictions['prefix_length']):
        print("\nWARNING: Mismatch in prediction counts!")
        print(f"Expected: {len(predictions['prefix_length'])}")
        print(f"Found: {len(existing_preds)}")
else:
    existing_preds = pd.DataFrame({'prefix_length': predictions['prefix_length']})

# Add outcome predictions
results_df = existing_preds.copy()
results_df['outcome_true'] = predictions['outcome_true']
results_df['outcome_pred'] = predictions['outcome_pred']
# Add dummy columns for activity prediction (since this is outcome-only model)
results_df['activity_true'] = -1  # Use -1 to indicate N/A
results_df['activity_pred'] = -1

# Standardize predictions
# results_df = standardize_predictions(predictions, trace_lengths=False)

# Merge with existing predictions based on case_id and prefix_length
# merged_df = pd.merge(
#     existing_preds,
#     results_df[['case_id', 'prefix_length', 'outcome_true', 'outcome_pred'] + 
#                     [col for col in results_df.columns if 'outcome_prob_' in col]],
#     on=['case_id', 'prefix_length'],
#     how='outer'
# )

# Add probability columns for each class
outcome_probs = np.array(predictions['outcome_probs'])
for i in range(outcome_probs.shape[1]):
    results_df[f'outcome_prob_{i}'] = outcome_probs[:, i]

# Save updated predictions to CSV
results_df.to_csv(f"{output_dir}/predictions_outcome.csv", index=False)
print(f"Updated predictions saved to {output_dir}/predictions_outcome.csv")

# # Collect all predictions and true labels
# all_preds = []
# all_labels = []
# all_prefix_lengths = []
# all_probs = []

#  # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Path to your saved model
# model_path = 'datasets/'+log+'/outcome_finetuned_model'

# # Load the tokenizer
# tokenizer = BertTokenizer.from_pretrained(model_path)

# # Load the model 
# model = BertForSequenceClassification.from_pretrained(model_path)
# model.to(device)

# # Load the label map for interpreting outputs
# import json
# with open(f'{model_path}/label_map.json', 'r') as f:
#     label_map = json.load(f)
#     id_to_label = {int(idx): outcome for outcome, idx in label_map.items()}

# model.eval()
# with torch.no_grad():
#     for batch in tqdm(test_loader, desc="Evaluating on test set"):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         position_ids = batch['position_ids'].to(device)
#         labels = batch['labels'].to(device)

#         # Get prefix lengths - count non-pad tokens in each sample
#         prefix_lengths = batch['original_prefix_length'].cpu().numpy()

#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#         )

#         logits = outputs.logits
#         probabilities = F.softmax(logits, dim=1)
#         predictions = torch.argmax(logits, dim=-1)

#         all_preds.extend(predictions.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#         all_prefix_lengths.extend(prefix_lengths)
#         all_probs.extend(probabilities.cpu().numpy())

# print_outcome_class_distribution(all_prefix_lengths, all_labels)

# # Calculate overall metrics
# accuracy = accuracy_score(all_labels, all_preds)
# macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
# precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
# recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

# # Convert to one-hot for ROC AUC
# one_hot_labels = np.zeros((len(all_labels), num_labels))
# for i, label in enumerate(all_labels):
#     one_hot_labels[i, label] = 1

# try:
#     roc_auc = roc_auc_score(one_hot_labels, all_probs, multi_class='ovr')
# except:
#     roc_auc = float('nan')  # In case of errors (e.g., single class)

# # Group by prefix length
# prefix_length_metrics = {}
# unique_lengths = sorted(set(all_prefix_lengths))

# for length in unique_lengths:
#     indices = [i for i, p_len in enumerate(all_prefix_lengths) if p_len == length]
#     if len(indices) == 0:
#         continue
        
#     length_preds = [all_preds[i] for i in indices]
#     length_labels = [all_labels[i] for i in indices]
#     length_probs = [all_probs[i] for i in indices]
    
#     # Calculate metrics for this length
#     try:
#         length_accuracy = accuracy_score(length_labels, length_preds)
#         length_f1 = f1_score(length_labels, length_preds, average='macro', zero_division=0)
#         length_precision = precision_score(length_labels, length_preds, average='macro', zero_division=0)
#         length_recall = recall_score(length_labels, length_preds, average='macro', zero_division=0)
        
#         # Convert to one-hot for ROC AUC
#         length_one_hot = np.zeros((len(length_labels), num_labels))
#         for i, label in enumerate(length_labels):
#             length_one_hot[i, label] = 1
            
#         try:
#             length_roc_auc = roc_auc_score(length_one_hot, length_probs, multi_class='ovr')
#         except:
#             length_roc_auc = float('nan')
            
#         prefix_length_metrics[length] = {
#             'Accuracy': length_accuracy,
#             'F1': length_f1,
#             'Precision': length_precision,
#             'Recall': length_recall,
#             'ROC_AUC': length_roc_auc,
#             'NumSamples': len(indices)
#         }
#     except:
#         # Skip if metrics calculation fails (e.g., single class)
#         continue

# # Save metrics to file
# save_path = 'datasets/'+log+'/outcome_finetuned_model'
# with open(f'{save_path}/outcome_metrics.txt', 'w') as f:
#     f.write("Overall Metrics:\n")
#     f.write(f"Accuracy: {accuracy:.4f}\n")
#     f.write(f"F1: {macro_f1:.4f}\n")
#     f.write(f"Precision: {precision:.4f}\n")
#     f.write(f"Recall: {recall:.4f}\n\n")
#     f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    
#     f.write("Metrics per prefix length:\n")
#     f.write("Length;Accuracy;F1;Precision;Recall;ROC_AUC;NumSamples\n")
    
#     for length in sorted(prefix_length_metrics.keys()):
#         metrics = prefix_length_metrics[length]
#         f.write(f"{length};{metrics['Accuracy']:.4f};{metrics['F1']:.4f};")
#         f.write(f"{metrics['Precision']:.4f};{metrics['Recall']:.4f};")
#         f.write(f"{metrics['ROC_AUC']:.4f};{metrics['NumSamples']}\n")
# print(f"Metrics saved to {save_path}/outcome_metrics.txt")

# # Generate classification report
# report = classification_report(
#     all_labels, all_preds, target_names=unique_outcomes
# )
# print("Classification Report on Test Set:")
# print(report)

# # Generate confusion matrix
# conf_matrix = confusion_matrix(all_labels, all_preds)
# print("Confusion Matrix on Test Set:")
# print(conf_matrix)

# # Save the fine-tuned model and tokenizer
# model.save_pretrained('outcome_finetuned_model')
# tokenizer.save_pretrained('outcome_finetuned_model')

# # Save the label map
# with open('datasets/'+log+'/outcome_finetuned_model/label_map.json', 'w') as f:
#     json.dump(label_map, f)

# # Example of using the fine-tuned model for inference
# # Load the model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('datasets/'+log+'/outcome_finetuned_model')
# model = BertForSequenceClassification.from_pretrained(
#     'datasets/'+log+'/outcome_finetuned_model'
# )
# model.to(device)

# # Load the label map
# with open('datasets/'+log+'/outcome_finetuned_model/label_map.json', 'r') as f:
#     label_map = json.load(f)
# id_to_label = {int(idx): outcome for outcome, idx in label_map.items()}

# # Prepare the input
# prefix = ['Contact - Aankoop/verkoop', 'Funnel - Offerte acceptatie']
# input_text = ' [SEP] '.join(prefix)
# encoding = tokenizer(
#     input_text,
#     add_special_tokens=True,
#     truncation=True,
#     max_length=180,
#     return_tensors='pt',
#     padding='max_length',
# )

# input_ids = encoding['input_ids'].to(device)
# attention_mask = encoding['attention_mask'].to(device)

# # Make prediction
# model.eval()
# with torch.no_grad():
#     outputs = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#     )
#     logits = outputs.logits
#     predicted_class_id = logits.argmax().item()
#     predicted_outcome = id_to_label[predicted_class_id]

# print(f"Predicted outcome: {predicted_outcome}")
