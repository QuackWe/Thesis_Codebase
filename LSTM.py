import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
from MTLFormer.encoding_mortgage import df_mortgages

# Assuming the processed data is stored in `processed_df`
processed_df = df_mortgages


# Step 1: Encode Activities as Integers
label_encoder = LabelEncoder()
processed_df['ActivityEncoded'] = label_encoder.fit_transform(processed_df['Activity'])
num_classes = processed_df['ActivityEncoded'].nunique()

# Step 2: Calculate Time Differences (seconds between events)
processed_df['TimeDiff'] = processed_df.groupby('CaseID')['Timestamp'].diff().dt.total_seconds().fillna(0)


# Step 3: Prepare Sequences for LSTM Input
class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.cases = df['CaseID'].unique()

        # Set maxlen dynamically based on the longest trace in the dataset
        self.maxlen = df.groupby('CaseID').size().max()

    def __getitem__(self, idx):
        case_id = self.cases[idx]
        case_data = self.df[self.df['CaseID'] == case_id]

        # Get activities and pad if necessary
        activities = case_data['ActivityEncoded'].values
        activities = np.pad(activities, (self.maxlen - len(activities), 0), 'constant')

        # Get time differences and pad if necessary
        times = case_data['TimeDiff'].values
        times = np.pad(times, (self.maxlen - len(times), 0), 'constant')

        # Target activity and time for next event prediction
        target_activity = np.append(activities[1:], 0)
        target_time = np.append(times[1:], 0)

        return (torch.tensor(activities, dtype=torch.long),
                torch.tensor(times, dtype=torch.float32),
                torch.tensor(target_activity, dtype=torch.long),
                torch.tensor(target_time, dtype=torch.float32))

    def __len__(self):
        return len(self.cases)


# Hyperparameters
# maxlen = 1  # max sequence length; adjust based on your data
batch_size = 32

# Create the dataset and dataloader
dataset = CustomDataset(processed_df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=100, dropout=0.2):
        super(LSTMModel, self).__init__()
        # Shared LSTM layer
        self.embedding = nn.Embedding(num_classes, num_features)  # Embedding for activities
        self.lstm1 = nn.LSTM(num_features + 1, hidden_size, batch_first=True)

        # Separate layers for activity and time prediction
        self.lstm_activity = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_time = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Output layers
        self.fc_activity = nn.Linear(hidden_size, num_classes)
        self.fc_time = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, activities, times):
        # Embed activity sequences
        act_emb = self.embedding(activities)
        x = torch.cat([act_emb, times.unsqueeze(-1)], dim=-1)

        # Shared LSTM
        x, _ = self.lstm1(x)
        x = self.dropout(x[:, -1, :])  # Take the last output

        # Activity prediction path
        act_out, _ = self.lstm_activity(x.unsqueeze(1))
        act_out = self.fc_activity(act_out[:, -1, :])

        # Time prediction path
        time_out, _ = self.lstm_time(x.unsqueeze(1))
        time_out = self.fc_time(time_out[:, -1, :])

        return act_out, time_out


# Initialize model, loss functions, and optimizer
num_features = 10  # embedding size for activities
model = LSTMModel(num_features, num_classes, hidden_size=100, dropout=0.2)

criterion_activity = nn.CrossEntropyLoss()
criterion_time = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for activities, times, target_activity, target_time in dataloader:
        optimizer.zero_grad()

        # Reshape act_pred and target_activity for cross-entropy
        # target_activity = target_activity.view(-1)  # Flatten to (batch_size * maxlen,)

        # Forward pass
        act_pred, time_pred = model(activities, times)
        # print(act_pred.shape, time_pred.shape)
        # print(num_classes)
        # # Flatten predictions and targets
        # batch_size, num_classes = act_pred.shape
        # act_pred = act_pred.view(-1, num_classes)  # Flatten to (batch_size * maxlen, num_classes)

        # Reshape act_pred for loss calculation
        act_pred = act_pred.view(-1, num_classes)  # Flatten to (batch_size * maxlen, num_classes)

        # Convert target_activity to class indices if it is one-hot or probabilities
        target_activity = torch.argmax(target_activity, dim=1)  # Shape: (batch_size * maxlen,)

        # Ensure target_activity is flattened
        target_activity = target_activity.view(-1)  # Final shape: (batch_size * maxlen,)

        # Compute loss
        loss_activity = criterion_activity(act_pred, target_activity)
        loss_time = criterion_time(time_pred, target_time)

        # Combine losses and backward pass
        loss = loss_activity + loss_time
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')
