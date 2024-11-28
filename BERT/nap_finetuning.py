import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertModel,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.modeling_bert import BertPooler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Parameters and file paths
DATA_FILE = 'preprocessed_prefixes.csv'
MAM_MODEL_PATH = 'mam_pretrained_model'
FINE_TUNED_MODEL_PATH = 'next_activity_finetuned_model'
LABEL_MAP_PATH = f'{FINE_TUNED_MODEL_PATH}/label_map.json'
BATCH_SIZE = 16
NUM_EPOCHS = 1
MAX_LEN = 128
LEARNING_RATE = 5e-5
TEST_SIZE = 0.1  # 10% of data for testing
VAL_SIZE = 0.1   # 10% of data for validation (from remaining 90%)

# Load the tokenizer and pre-trained MAM model
tokenizer = BertTokenizer.from_pretrained(MAM_MODEL_PATH)
mam_model = BertForMaskedLM.from_pretrained(MAM_MODEL_PATH)

# Load the configuration from the MAM model
config = mam_model.config

# Load the data
df = pd.read_csv(DATA_FILE)

# Create label mappings
unique_activities = df['MaskedActivity'].unique().tolist()
label_map = {activity: idx for idx, activity in enumerate(unique_activities)}
id_to_label = {idx: activity for activity, idx in label_map.items()}
num_labels = len(label_map)
config.num_labels = num_labels

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
train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=42)

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
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

# Create datasets and data loaders
train_dataset = NextActivityDataset(train_df, tokenizer, label_map, max_len=MAX_LEN)
val_dataset = NextActivityDataset(val_df, tokenizer, label_map, max_len=MAX_LEN)
test_dataset = NextActivityDataset(test_df, tokenizer, label_map, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = NUM_EPOCHS * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training loop
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
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

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Training completed. Average Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_examples = 0

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

    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = total_correct / total_examples
    print(f"Epoch {epoch+1}: Validation completed. Average Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

# Evaluation on the test set
model.eval()
total_test_loss = 0
total_correct = 0
total_examples = 0
all_preds = []
all_labels = []

with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="Evaluating on Test Set")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = batch['position_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        total_test_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)

        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_test_loss = total_test_loss / len(test_loader)
test_accuracy = total_correct / total_examples
print(f"Test Set Evaluation: Average Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

# Get unique labels in the validation set
val_unique_labels = np.unique(all_labels)

# Map label IDs back to activity names
val_unique_activities = [id_to_label[label_id] for label_id in val_unique_labels]

# Generate classification report
report = classification_report(
    all_labels, all_preds, labels=val_unique_labels, target_names=val_unique_activities
)
print("Classification Report on Test Set:")
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix on Test Set:")
print(conf_matrix)

# Save the fine-tuned model and tokenizer
model.save_pretrained(FINE_TUNED_MODEL_PATH)
tokenizer.save_pretrained(FINE_TUNED_MODEL_PATH)

# Save the label map (already saved, but ensuring it's up-to-date)
with open(LABEL_MAP_PATH, 'w') as f:
    json.dump(label_map, f)

# Example usage: Predicting the next activity for a given prefix
def predict_next_activity(prefix_activities):
    input_text = ' [SEP] '.join(prefix_activities)
    encoding = tokenizer(
        input_text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    position_ids = torch.arange(MAX_LEN, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        predicted_activity = id_to_label[predicted_class_id]

    return predicted_activity

# Example prediction
prefix = ['Contact - Aankoop/verkoop', 'Funnel - Offerte acceptatie']
predicted_activity = predict_next_activity(prefix)
print(f"Predicted next activity: {predicted_activity}")
