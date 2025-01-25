import torch
from torch.utils.data import Dataset

class TraceDataset(Dataset):
    def __init__(self, dataset):
        self.traces = torch.tensor(dataset['Trace'].tolist(), dtype=torch.long)
        self.times = torch.tensor(dataset['Times'].tolist(), dtype=torch.float)
        self.next_activities = torch.tensor(dataset['NextActivity'].tolist(), dtype=torch.long)
        self.masks = torch.tensor(dataset['AttentionMask'].tolist(), dtype=torch.long)
        self.outcomes = torch.tensor(dataset['Outcome'].tolist(), dtype=torch.long)
        self.customer_types = torch.tensor(dataset['CustomerType'].tolist(), dtype=torch.float)
        # Initialize loop_features as None, will be set later
        self.loop_features = None

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        item = {
            'trace': self.traces[idx],
            'times': self.times[idx],
            'next_activity': self.next_activities[idx],
            'mask': self.masks[idx],
            'outcome': self.outcomes[idx],
            'customer_type': self.customer_types[idx]
        }
        # Add loop_features if they exist
        if self.loop_features is not None:
            item['loop_features'] = self.loop_features[idx]

        return item
