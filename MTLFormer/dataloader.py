import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from sys import argv


# Directory setup
log = argv[1]
dataset_path = os.path.join('datasets', log)

# Load data from files
X = torch.load(os.path.join(dataset_path, "X.pt"))
y_activity = torch.load(os.path.join(dataset_path, "y_activity.pt"))
y_outcome = torch.load(os.path.join(dataset_path, "y_outcome.pt"))
y_next_time = torch.load(os.path.join(dataset_path, "y_next_time.pt"))
y_remaining_time = torch.load(os.path.join(dataset_path, "y_remaining_time.pt"))

# Set batch size
batch_size = 64


class EventLogDataset(Dataset):
    def __init__(self, X, y_activity, y_outcome, y_next_time, y_remaining_time):
        self.X = X
        self.y_activity = y_activity
        self.y_outcome = y_outcome
        self.y_next_time = y_next_time
        self.y_remaining_time = y_remaining_time

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'sequence': self.X[idx],
            'next_activity': self.y_activity[idx],
            'final_outcome': self.y_outcome[idx],
            'next_event_time': self.y_next_time[idx],
            'remaining_time': self.y_remaining_time[idx]
        }


# Split data into training and test sets (80% train/val, 20% test)
X_trainval, X_test, y_activity_trainval, y_activity_test, y_outcome_trainval, y_outcome_test, y_next_time_trainval, y_next_time_test, y_remaining_time_trainval, y_remaining_time_test = train_test_split(
    X, y_activity, y_outcome, y_next_time, y_remaining_time, test_size=0.2, random_state=42)

# Split train data into training and validation sets (80% train, 20% val)
X_train, X_val, y_activity_train, y_activity_val, y_outcome_train, y_outcome_val, y_next_time_train, y_next_time_val, y_remaining_time_train, y_remaining_time_val = train_test_split(
    X_trainval, y_activity_trainval, y_outcome_trainval, y_next_time_trainval, y_remaining_time_trainval, test_size=0.2, random_state=42)


# Create datasets
train_dataset = EventLogDataset(X_train, y_activity_train, y_outcome_train, y_next_time_train, y_remaining_time_train)
val_dataset = EventLogDataset(X_val, y_activity_val, y_outcome_val, y_next_time_val, y_remaining_time_val)
test_dataset = EventLogDataset(X_test, y_activity_test, y_outcome_test, y_next_time_test, y_remaining_time_test)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)