import json
import numpy as np
import pandas as pd
from sys import argv
import torch
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

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
log = argv[1]

# Load the tokenizer and pre-trained MAM model
tokenizer = BertTokenizer.from_pretrained('datasets/'+log+'/mam_pretrained_model')
mam_model = BertForMaskedLM.from_pretrained('datasets/'+log+'/mam_pretrained_model')

# Load the configuration from the MAM model
config = mam_model.config

# Load and preprocess the data
event_log_file = 'datasets/'+log+'/'+log+'_processed.csv'
df = pd.read_csv(event_log_file, parse_dates=['Timestamp'])
df = df.sort_values(by=['CaseID', 'Timestamp'])

# Group by CaseID to form traces
grouped = df.groupby('CaseID')

data = []
for case_id, group in grouped:
    activities = group['Activity'].tolist()
    final_outcome = group['FinalOutcome'].iloc[0]  # Assuming FinalOutcome is the same for all events in a case

    # Create prefixes
    for i in range(1, len(activities) + 1):
        prefix = activities[:i]
        data.append({'Prefix': prefix, 'FinalOutcome': final_outcome})

# Create DataFrame
dataset_df = pd.DataFrame(data)

# Get the list of unique outcomes from the 'FinalOutcome' column
unique_outcomes = dataset_df['FinalOutcome'].unique().tolist()
num_labels = len(unique_outcomes)
config.num_labels = num_labels

# Create label mappings
label_map = {outcome: idx for idx, outcome in enumerate(unique_outcomes)}
id_to_label = {idx: outcome for outcome, idx in label_map.items()}

# Save label mappings for future use
with open('datasets/'+log+'/outcome_label_map.json', 'w') as f:
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

# Split the dataset into train, validation, and test sets
# First split into train+val and test sets
train_val_df, test_df = train_test_split(
    dataset_df,
    test_size=0.2,
    random_state=42,
    stratify=dataset_df['FinalOutcome'],
)

# Then split train_val_df into train and validation sets
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1,
    random_state=42,
    stratify=train_val_df['FinalOutcome'],
)

# Save the datasets to CSV
train_df.to_csv('datasets/'+log+'/outcome_train.csv', index=False)
val_df.to_csv('datasets/'+log+'/outcome_val.csv', index=False)
test_df.to_csv('datasets/'+log+'/outcome_test.csv', index=False)

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
        }


# Create datasets and DataLoaders
train_dataset = OutcomePredictionDataset('datasets/'+log+'/outcome_train.csv', tokenizer, label_map)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = OutcomePredictionDataset('datasets/'+log+'/outcome_val.csv', tokenizer, label_map)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

test_dataset = OutcomePredictionDataset('datasets/'+log+'/outcome_test.csv', tokenizer, label_map)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Set up the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10
num_training_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Fine-tuning loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]"
    )
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

    avg_train_loss = total_train_loss / len(train_loader)
    print(
        f"Epoch {epoch + 1} Training completed. Average Loss: {avg_train_loss:.4f}"
    )

    # Validation phase
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_examples = 0

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

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_correct / total_examples
    print(
        f"Epoch {epoch + 1} Validation completed. Average Loss: {avg_val_loss:.4f}, "
        f"Accuracy: {val_accuracy:.4f}"
    )

# Evaluation on the test set
# Collect all predictions and true labels
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating on test set"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = batch['position_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate classification report
report = classification_report(
    all_labels, all_preds, target_names=unique_outcomes
)
print("Classification Report on Test Set:")
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix on Test Set:")
print(conf_matrix)

# Save the fine-tuned model and tokenizer
model.save_pretrained('outcome_finetuned_model')
tokenizer.save_pretrained('outcome_finetuned_model')

# Save the label map
with open('datasets/'+log+'/outcome_finetuned_model/label_map.json', 'w') as f:
    json.dump(label_map, f)

# Example of using the fine-tuned model for inference
# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('datasets/'+log+'/outcome_finetuned_model')
model = BertForSequenceClassification.from_pretrained(
    'datasets/'+log+'/outcome_finetuned_model'
)
model.to(device)

# Load the label map
with open('datasets/'+log+'/outcome_finetuned_model/label_map.json', 'r') as f:
    label_map = json.load(f)
id_to_label = {int(idx): outcome for outcome, idx in label_map.items()}

# Prepare the input
prefix = ['Contact - Aankoop/verkoop', 'Funnel - Offerte acceptatie']
input_text = ' [SEP] '.join(prefix)
encoding = tokenizer(
    input_text,
    add_special_tokens=True,
    truncation=True,
    max_length=128,
    return_tensors='pt',
    padding='max_length',
)

input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

# Make prediction
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_outcome = id_to_label[predicted_class_id]

print(f"Predicted outcome: {predicted_outcome}")
