import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    BertTokenizer,
    BertModel,
    BertForMaskedLM,
    AdamW,
    BertPreTrainedModel,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
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
config = mam_model.config  # Load configuration

# Load and preprocess the dataset
df = pd.read_csv(DATA_FILE)

# Create label mappings
unique_activities = df['MaskedActivity'].unique().tolist()
label_map = {activity: idx for idx, activity in enumerate(unique_activities)}
id_to_label = {idx: activity for activity, idx in label_map.items()}
num_labels = len(label_map)
config.num_labels = num_labels  # Update the number of labels in the config
class_weights = np.ones(num_labels, dtype=np.float32) # Initialize class_weights with ones

# Save label mappings
if not os.path.exists(FINE_TUNED_MODEL_PATH):
    os.makedirs(FINE_TUNED_MODEL_PATH)
with open(LABEL_MAP_PATH, 'w') as f:
    json.dump(label_map, f)


# Custom Model with Global Average Pooling (GAP)
class CustomBertForNextActivityPrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)  # Use the BERT encoder
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)  # GAP layer
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)  # Dropout layer
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # Classification layer

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

        # Apply Global Average Pooling (GAP)
        gap_input = hidden_states.permute(0, 2, 1)  # Shape: [batch_size, hidden_size, seq_len]
        pooled_output = self.global_avg_pooling(gap_input).squeeze(-1)  # Shape: [batch_size, hidden_size]

        # Apply dropout
        dropped_out = self.dropout(pooled_output)  # Shape: [batch_size, hidden_size]

        # Pass through the classifier
        logits = self.classifier(dropped_out)  # Shape: [batch_size, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]  # Skip past hidden states and attentions
            return ((loss,) + output) if loss is not None else output

        return {'loss': loss, 'logits': logits, 'hidden_states': outputs.hidden_states,
                'attentions': outputs.attentions}


# Dataset Class for Next Activity Prediction
class NextActivityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_map, max_len=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the prefix and masked activity
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

        # Generate position_ids
        position_ids = torch.arange(self.max_len, dtype=torch.long)  # Shape: [max_len]

        # Get label ID
        label_id = self.label_map[next_activity]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': torch.tensor(label_id, dtype=torch.long)
        }


# Prepare data for training and validation
train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=42)

train_dataset = NextActivityDataset(train_df, tokenizer, label_map, max_len=MAX_LEN)
val_dataset = NextActivityDataset(val_df, tokenizer, label_map, max_len=MAX_LEN)
test_dataset = NextActivityDataset(test_df, tokenizer, label_map, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Initialize and Prepare Model
model = CustomBertForNextActivityPrediction(config)
model.bert.load_state_dict(mam_model.bert.state_dict(), strict=False)  # Load MAM weights
model.init_weights()
model.to(device)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# Compute class weights for balanced loss
# Map 'MaskedActivity' to label IDs in the training data
y_series = train_dataset.data['MaskedActivity'].map(label_map)
y = y_series.values.astype(int)

# Get unique classes present in y
present_classes = np.unique(y)

# Compute class weights for classes present in y
class_weights_present = compute_class_weight(
    class_weight='balanced',
    classes=present_classes,
    y=y
)

# Initialize class_weights with ones
class_weights = np.ones(num_labels, dtype=np.float32)

# Update weights for classes present in y
for idx, class_id in enumerate(present_classes):
    class_weights[class_id] = class_weights_present[idx]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)


# Training and Validation Loop
for epoch in range(NUM_EPOCHS):
    # Training Phase
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
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
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix({'loss': total_train_loss / (progress_bar.n + 1)})

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Training completed. Average Loss: {avg_train_loss:.4f}")

    # Validation Phase
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_ids = batch['position_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            logits = outputs["logits"]
            loss = loss_fn(logits, labels)

            total_val_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = total_correct / total_examples
    print(f"Epoch {epoch+1}: Validation completed. Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
    scheduler.step(avg_val_loss)

    # Optionally, print the current learning rate
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print(f"Current Learning Rate after Epoch {epoch + 1}: {current_lr}")

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

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs["logits"]
        loss = loss_fn(logits, labels)

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

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        predicted_class_id = logits.argmax().item()
        predicted_activity = id_to_label[predicted_class_id]

    return predicted_activity

# Example prediction
prefix = ['Contact - Aankoop/verkoop', 'Funnel - Offerte acceptatie']
predicted_activity = predict_next_activity(prefix)
print(f"Predicted next activity: {predicted_activity}")
