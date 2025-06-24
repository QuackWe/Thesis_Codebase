import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertModel,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.modeling_bert import BertPooler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import argparse
import time


# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True, help='Base output directory')
parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
args = parser.parse_args()
log = args.log
output_dir = args.output_dir


# Parameters and file paths
DATA_FILE_TRAINVAL = output_dir+'/preprocessed_prefixes_train-val.csv'
DATA_FILE_TEST = output_dir+'/preprocessed_prefixes_test.csv'
MAM_MODEL_PATH = output_dir+'/mam_pretrained_model'
FINE_TUNED_MODEL_PATH = output_dir+'/next_activity_finetuned_model'
LABEL_MAP_PATH = f'{FINE_TUNED_MODEL_PATH}/label_map.json'
BATCH_SIZE = 16
NUM_EPOCHS = 1
# MAX_LEN = 180
LEARNING_RATE = 5e-5
VAL_SIZE = 0.1   # 10% of data for validation (from remaining 90%)
# Add these near the top with other parameters
MAX_SAMPLES = 20  # Small number for testing
USE_SAMPLE_LIMIT = False  # Flag to toggle sample limiting


# Load the tokenizer and pre-trained MAM model
tokenizer = BertTokenizer.from_pretrained(MAM_MODEL_PATH)
mam_model = BertForMaskedLM.from_pretrained(MAM_MODEL_PATH)

# Load the configuration from the MAM model
config = mam_model.config

# Load the data
train_val_df = pd.read_csv(DATA_FILE_TRAINVAL)
test_df = pd.read_csv(DATA_FILE_TEST)

# Create label mappings
unique_activities = train_val_df['MaskedActivity'].unique().tolist()
label_map = {activity: idx for idx, activity in enumerate(unique_activities)}
id_to_label = {idx: activity for activity, idx in label_map.items()}
num_labels = len(label_map)
config.num_labels = num_labels

# Modify the data loading section (after loading the CSVs)
if USE_SAMPLE_LIMIT:
    train_val_df = train_val_df.head(MAX_SAMPLES)
    # test_df = test_df.head(MAX_SAMPLES // 2)  # Using fewer test samples
    print(f"Limited dataset to {len(train_val_df)} train/val samples and {len(test_df)} test samples")

# Save label mappings
if not os.path.exists(FINE_TUNED_MODEL_PATH):
    os.makedirs(FINE_TUNED_MODEL_PATH)
with open(LABEL_MAP_PATH, 'w') as f:
    json.dump(label_map, f)

# Initialize BertModel without the pooling layer and add the pooler layer manually
bert_model = BertModel(config, add_pooling_layer=False)
bert_model.pooler = BertPooler(config)
bert_model.pooler.apply(bert_model._init_weights)

# Load pre-trained weights into bert_model
bert_model.load_state_dict(mam_model.bert.state_dict(), strict=False)

# Initialize the sequence classification model and replace its BertModel
model = BertForSequenceClassification(config)
model.bert = bert_model

# Move the model to the device
model.to(device)

# Verify the pooler layer is present
print("Is the pooler layer present in model.bert?", hasattr(model.bert, 'pooler'))

# Split the data into train, validation, and test sets
train_df, val_df = train_test_split(train_val_df, test_size=VAL_SIZE, random_state=42)

# Define the dataset class
class NextActivityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_map, max_len=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prefix = eval(self.data.loc[idx, 'Prefix'])  # Convert string to list
        prefix_length = len(prefix)  # Store the original prefix length
        next_activity = self.data.loc[idx, 'MaskedActivity']

        # Prepare input text
        input_text = ' [SEP] '.join(prefix)

        # Tokenize input
        encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        position_ids = torch.arange(self.max_len, dtype=torch.long)

        # Get label ID
        label_id = self.label_map[next_activity]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': torch.tensor(label_id, dtype=torch.long),
            'original_prefix_length': torch.tensor(prefix_length, dtype=torch.long)
        }

# Create datasets and data loaders
train_dataset = NextActivityDataset(train_df, tokenizer, label_map) #, max_len=MAX_LEN)
val_dataset = NextActivityDataset(val_df, tokenizer, label_map) #, max_len=MAX_LEN)
test_dataset = NextActivityDataset(test_df, tokenizer, label_map) #, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Add this before generating predictions DEBUG
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Number of batches in test loader: {len(test_loader)}")

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = NUM_EPOCHS * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

print("\nStarting training phase...")
training_start_time = time.time()

# Modify training loop:
# Fine-tuning loop
total_train_time = 0
total_val_time = 0
epoch_train_start = time.time()

# Training loop
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
    epoch_start_time = time.time()

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = batch['position_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix({'loss': total_train_loss / (progress_bar.n + 1)})

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
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_ids = batch['position_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_val_loss += loss.item()
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
print(f"Average training time per epoch: {total_train_time/NUM_EPOCHS:.2f}s")
print(f"Average validation time per epoch: {total_val_time/NUM_EPOCHS:.2f}s")


# Evaluation on the test set
# Collect all predictions in standardized format
predictions = {
    'prefix_length': [],
    'case_id': [],
    'activity_true': [],
    'activity_pred': [],
    'activity_probs': []
}

print("\nStarting evaluation phase...")
eval_start_time = time.time()
model.eval()
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
        predictions['activity_true'].extend(labels.cpu().numpy())
        predictions['activity_pred'].extend(torch.argmax(logits, dim=1).cpu().numpy())
        predictions['activity_probs'].extend(probs.cpu().numpy())

eval_time = time.time() - eval_start_time
print(f"\nTotal evaluation time: {eval_time:.2f}s")

# Save timing metrics to file
timing_metrics = {
    'total_training_time': training_time,
    'avg_training_time_per_epoch': total_train_time/NUM_EPOCHS,
    'avg_validation_time_per_epoch': total_val_time/NUM_EPOCHS,
    'total_evaluation_time': eval_time
}

with open(f"{output_dir}/timing_metrics_nap.json", 'w') as f:
    json.dump(timing_metrics, f, indent=4)
print(f"Timing metrics saved to {output_dir}/timing_metrics_nap.json")
# Add this before generating predictions
print(f"Test dataset size: {len(test_dataset)}")
print(f"Number of batches in test loader: {len(test_loader)}")

# Add this after collecting predictions
print(f"Number of collected predictions: {len(predictions['prefix_length'])}")


# Convert to DataFrame
results_df = pd.DataFrame({
    'prefix_length': predictions['prefix_length'],
    'activity_true': predictions['activity_true'],
    'activity_pred': predictions['activity_pred']
})

# Add probability columns for each class
activity_probs = np.array(predictions['activity_probs'])
for i in range(activity_probs.shape[1]):
    results_df[f'activity_prob_{i}'] = activity_probs[:, i]

# Add dummy columns for outcome prediction (since this is activity-only model)
results_df['outcome_true'] = -1  # Use -1 to indicate N/A
results_df['outcome_pred'] = -1
results_df['outcome_prob_0'] = np.nan
results_df['outcome_prob_1'] = np.nan

# Save predictions to CSV
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(f"{output_dir}/predictions_nap.csv", index=False)
print(f"Predictions saved to {output_dir}/predictions_nap.csv")

# model.eval()
# total_test_loss = 0
# total_correct = 0
# total_examples = 0
# all_preds = []
# all_labels = []

# with torch.no_grad():
#     progress_bar = tqdm(test_loader, desc="Evaluating on Test Set")
#     for batch in progress_bar:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         position_ids = batch['position_ids'].to(device)
#         labels = batch['labels'].to(device)

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
#         loss = outputs.loss
#         logits = outputs.logits

#         total_test_loss += loss.item()
#         predictions = torch.argmax(logits, dim=-1)

#         total_correct += (predictions == labels).sum().item()
#         total_examples += labels.size(0)

#         all_preds.extend(predictions.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# avg_test_loss = total_test_loss / len(test_loader)
# test_accuracy = total_correct / total_examples
# print(f"Test Set Evaluation: Average Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}")


# # Save metrics per prefix length
# all_prefix_lengths = []

# # Rerun inference to capture prefix lengths
# model.eval()
# with torch.no_grad():
#     for batch in tqdm(test_loader, desc="Collecting prefix lengths"):
#         # Get prefix length from attention mask (count non-zero elements)
#         prefix_lengths = batch['original_prefix_length'].cpu().numpy()
#         all_prefix_lengths.extend(prefix_lengths)

# # Group by prefix length
# prefix_length_metrics = {}
# unique_lengths = sorted(set(all_prefix_lengths))

# for length in unique_lengths:
#     indices = [i for i, p_len in enumerate(all_prefix_lengths) if p_len == length]
#     if len(indices) == 0:
#         continue
        
#     length_preds = [all_preds[i] for i in indices]
#     length_labels = [all_labels[i] for i in indices]
    
#     # Calculate metrics for this length
#     try:
#         length_accuracy = accuracy_score(length_labels, length_preds)
#         length_f1 = f1_score(length_labels, length_preds, average='macro')
#         length_precision = precision_score(length_labels, length_preds, average='macro')
#         length_recall = recall_score(length_labels, length_preds, average='macro')
        
#         # Convert to one-hot for ROC AUC (assuming you've saved probabilities)
#         # If you haven't saved probabilities, set ROC AUC to NaN
#         length_roc_auc = float('nan')
                
#         prefix_length_metrics[length] = {
#             'Accuracy': length_accuracy,
#             'F1': length_f1,
#             'Precision': length_precision,
#             'Recall': length_recall,
#             'ROC_AUC': length_roc_auc,
#             'NumSamples': len(indices)
#         }
#     except Exception as e:
#         print(f"Error calculating metrics for length {length}: {e}")
#         continue

# # Calculate overall metrics
# accuracy = accuracy_score(all_labels, all_preds)
# macro_f1 = f1_score(all_labels, all_preds, average='macro')
# precision = precision_score(all_labels, all_preds, average='macro')
# recall = recall_score(all_labels, all_preds, average='macro')
# roc_auc = float('nan')  # Set to NaN as we likely don't have probabilities

# # Save metrics to file
# save_path = f'datasets/{log}/nap_metrics.txt'
# with open(save_path, 'w') as f:
#     f.write("Overall Metrics:\n")
#     f.write(f"Accuracy: {accuracy:.4f}\n")
#     f.write(f"F1: {macro_f1:.4f}\n")
#     f.write(f"Precision: {precision:.4f}\n")
#     f.write(f"Recall: {recall:.4f}\n\n")
#     f.write(f"ROC AUC: {roc_auc}\n\n")
    
#     f.write("Metrics per prefix length:\n")
#     f.write("Length;Accuracy;F1;Precision;Recall;ROC_AUC;NumSamples\n")
    
#     for length in sorted(prefix_length_metrics.keys()):
#         metrics = prefix_length_metrics[length]
#         f.write(f"{length};{metrics['Accuracy']:.4f};{metrics['F1']:.4f};")
#         f.write(f"{metrics['Precision']:.4f};{metrics['Recall']:.4f};")
#         f.write(f"{metrics['ROC_AUC']:.4f};{metrics['NumSamples']}\n")

# print(f"Metrics saved to {save_path}")


# # Get unique labels in the validation set
# val_unique_labels = np.unique(all_labels)

# # Map label IDs back to activity names
# val_unique_activities = [id_to_label[label_id] for label_id in val_unique_labels]

# # Generate classification report
# report = classification_report(
#     all_labels, all_preds, labels=val_unique_labels, target_names=val_unique_activities
# )
# print("Classification Report on Test Set:")
# print(report)

# # Generate confusion matrix
# conf_matrix = confusion_matrix(all_labels, all_preds)
# print("Confusion Matrix on Test Set:")
# print(conf_matrix)

# # Save the fine-tuned model and tokenizer
# model.save_pretrained(FINE_TUNED_MODEL_PATH)
# tokenizer.save_pretrained(FINE_TUNED_MODEL_PATH)

# # Save the label map (already saved, but ensuring it's up-to-date)
# with open(LABEL_MAP_PATH, 'w') as f:
#     json.dump(label_map, f)

# # Example usage: Predicting the next activity for a given prefix
# def predict_next_activity(prefix_activities):
#     input_text = ' [SEP] '.join(prefix_activities)
#     encoding = tokenizer(
#         input_text,
#         add_special_tokens=True,
#         truncation=True,
#         max_length=MAX_LEN,
#         padding='max_length',
#         return_tensors='pt'
#     )

#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
#     position_ids = torch.arange(MAX_LEN, dtype=torch.long).unsqueeze(0).to(device)

#     model.eval()
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
#         logits = outputs.logits
#         predicted_class_id = logits.argmax().item()
#         predicted_activity = id_to_label[predicted_class_id]

#     return predicted_activity

# # Example prediction
# prefix = ['Contact - Aankoop/verkoop', 'Funnel - Offerte acceptatie']
# predicted_activity = predict_next_activity(prefix)
# print(f"Predicted next activity: {predicted_activity}")
