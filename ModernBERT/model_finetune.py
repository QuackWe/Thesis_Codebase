import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ModernBertModel, ModernBertConfig, ModernBertPreTrainedModel
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import csv
import torch.nn.functional as F
from collections import defaultdict
from pm4py.objects.log.importer.xes import importer as xes_importer
from torch.cuda import memory_stats
import GPUtil
import argparse

# ------------------------------
# Util Functions
# ------------------------------

# Load BPIC2017 dataset
def preprocess_bpic2017(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Extract activity names and prefixes (A_, O_, W_)
    df['Activity'] = df['concept:name'].astype(str)
    df['trace_id'] = df['case:concept:name'].astype(str)
    
    # Extract timestamp
    df['TimestampContact'] = pd.to_datetime(df['time:timestamp'])
    
    # Extract case attributes
    case_attributes = ['ApplicationType', 'LoanGoal', 'RequestedAmount']
    
    # Extract outcome (need to determine based on final states)
    # This is a simplified approach - you may need to adjust based on your specific data
    outcomes = []
    for trace_id, group in df.groupby('trace_id'):
        last_state = group.sort_values('TimestampContact')['Activity'].iloc[-1]
        
        # Simplified outcome determination
        outcome = 0  # Default (ongoing/other)
        if 'A_Accepted' in last_state or 'O_Accepted' in last_state:
            outcome = 1  # Approved
        elif 'A_Denied' in last_state or 'O_Refused' in last_state:
            outcome = 2  # Declined
        elif 'A_Cancelled' in last_state or 'O_Cancelled' in last_state:
            outcome = 3  # Canceled
            
        outcomes.extend([outcome] * len(group))
    
    df['outcome'] = outcomes
    
    return df

def compute_per_sample_losses(activity_logits, outcome_logits, activity_labels, outcome_labels):
    """Compute per-sample losses for prompt updates"""
    activity_criterion = nn.CrossEntropyLoss(reduction='none')
    outcome_criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Initialize loss tensors with zeros on the appropriate device
    device = activity_labels.device if activity_labels is not None else outcome_labels.device
    
    # Initialize empty tensors for both losses
    per_sample_activity_losses = torch.zeros(activity_labels.size(0) if activity_labels is not None else outcome_labels.size(0), device=device)
    per_sample_outcome_losses = torch.zeros(outcome_labels.size(0) if outcome_labels is not None else activity_labels.size(0), device=device)
    
    # Calculate activity losses if activity head is active
    if activity_logits is not None and activity_labels is not None:
        per_sample_activity_losses = activity_criterion(activity_logits, activity_labels)
    
    # Calculate outcome losses if outcome head is active
    if outcome_logits is not None and outcome_labels is not None:
        per_sample_outcome_losses = outcome_criterion(outcome_logits, outcome_labels)
    
    return per_sample_activity_losses, per_sample_outcome_losses

def visualize_prompt_evolution(prompt_manager, log_dir):
    """Visualize how prompts evolved during training"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    for ctype in range(prompt_manager.num_customer_types):
        if len(prompt_manager.prompt_history[ctype]['epochs']) == 0:
            continue
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Move all tensors to CPU and stack them
        activity_prompts = torch.stack([
            p.cpu() for p in prompt_manager.prompt_history[ctype]['activity_prompts']
        ])
        outcome_prompts = torch.stack([
            p.cpu() for p in prompt_manager.prompt_history[ctype]['outcome_prompts']
        ])
        
        # Plot activity prompt changes
        sns.heatmap(activity_prompts.mean(dim=1), ax=ax1, 
                    cmap='viridis', xticklabels=10, yticklabels=5)
        ax1.set_title(f'Activity Prompt Evolution for Customer Type {ctype}')
        ax1.set_ylabel('Epoch')
        
        # Plot outcome prompt changes
        sns.heatmap(outcome_prompts.mean(dim=1), ax=ax2, 
                    cmap='viridis', xticklabels=10, yticklabels=5)
        ax2.set_title(f'Outcome Prompt Evolution for Customer Type {ctype}')
        ax2.set_ylabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(f'{log_dir}/prompt_evolution_type_{ctype}.png')
        plt.close()

def calculate_normalization_params(df):
    """Calculate min and max time differences from the training data"""
    time_diffs = []
    for _, group in df.groupby('trace_id'):
        group = group.sort_values('TimestampContact')
        timestamps = group['TimestampContact'].tolist()
        for i in range(len(timestamps)-1):
            diff = (pd.to_datetime(timestamps[i+1]) - pd.to_datetime(timestamps[i])).total_seconds()
            time_diffs.append(diff)
    
    if not time_diffs:
        return 0, 1  # Return default values if no time differences
        
    # Log transform the time differences
    time_diffs = np.log1p(time_diffs)
    return np.min(time_diffs), np.max(time_diffs)


# ------------------------------
# Improved Dataset Class with Prefix Generation
# ------------------------------
class ProcessTraceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64, use_act_separators=True, sliding_window=True, window_stride=None, normalization_params=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_act_separators = use_act_separators
        self.sliding_window = sliding_window
        self.window_stride = window_stride if window_stride is not None else max_length // 2
        self.min_diff = normalization_params[0] if normalization_params else None
        self.max_diff = normalization_params[1] if normalization_params else None
        
        # Add debug counters
        self.prefix_counts = {}
        self.sliding_window_counts = {}
        self.total_samples_per_prefix = {}
        
        # Process traces and generate samples directly
        self.samples = []
        
        self.customer_type_encoder = LabelEncoder()
        self.customer_type_encoder.fit(df['type_of_customer'].unique())

        for trace_id, group in df.groupby('trace_id'):
            group = group.sort_values('TimestampContact')
            activities = group['Activity'].tolist()
            timestamps = group['TimestampContact'].tolist()
            outcome = group['outcome'].iloc[0]
            customer_type = group['type_of_customer'].iloc[0]
            customer_type_id = self.customer_type_encoder.transform([customer_type])[0]

            # Calculate time differences in seconds and normalize per trace
            time_diffs = []
            for i in range(len(timestamps)-1):
                diff = (pd.to_datetime(timestamps[i+1]) - pd.to_datetime(timestamps[i])).total_seconds()
                time_diffs.append(diff)
            
            # Normalize time differences using log-scale and provided parameters
            if time_diffs:
                time_diffs = np.log1p(time_diffs)
                if self.min_diff is not None and self.max_diff is not None:
                    # Use provided normalization parameters
                    if self.max_diff > self.min_diff:
                        time_diffs = (time_diffs - self.min_diff) / (self.max_diff - self.min_diff)
                    else:
                        time_diffs = np.zeros_like(time_diffs)
                else:
                    # Fallback to per-trace normalization if no parameters provided
                    min_diff = np.min(time_diffs)
                    max_diff = np.max(time_diffs)
                    if max_diff > min_diff:
                        time_diffs = (time_diffs - min_diff) / (max_diff - min_diff)
                    else:
                        time_diffs = np.zeros_like(time_diffs)

            # Generate all prefixes (minimum length 1)
            for i in range(1, len(activities)):
                prefix = activities[:i]
                next_activity = activities[i]
                prefix_time_diffs = time_diffs[:i-1]  # Time diffs up to current point
                prefix_len = len(prefix)
                
                # Add padding for time differences if needed
                padded_time_diffs = np.zeros(self.max_length)  # -1 because we don't need time diff for first token
                if len(prefix_time_diffs) > 0:
                    # Ensure we don't exceed array bounds
                    n_diffs = min(len(prefix_time_diffs), self.max_length - 1)
                    padded_time_diffs[1:n_diffs+1] = prefix_time_diffs[:n_diffs]
                
                # Track regular prefix samples
                if prefix_len not in self.prefix_counts:
                    self.prefix_counts[prefix_len] = 0
                    self.sliding_window_counts[prefix_len] = 0
                    self.total_samples_per_prefix[prefix_len] = 0
                self.prefix_counts[prefix_len] += 1
                
                # Format the prefix text with SEP tokens
                prefix_text = ' [SEP] '.join(prefix)
                
                # First, tokenize without truncation to get full sequence
                full_encoding = self.tokenizer(
                    prefix_text,
                    add_special_tokens=True,
                    truncation=False
                )
                
                input_ids_full = full_encoding['input_ids']
                
                if self.sliding_window and len(input_ids_full) > self.max_length:
                    # Count actual windows that will be created
                    num_windows = len(range(0, len(input_ids_full) - self.max_length + 1, self.window_stride))
                    self.sliding_window_counts[prefix_len] += num_windows
                    self.total_samples_per_prefix[prefix_len] += num_windows
                else:
                    self.total_samples_per_prefix[prefix_len] += 1

            # # Print actual totals
            # print("\nActual samples per prefix length:")
            # for prefix_len in sorted(self.total_samples_per_prefix.keys()):
            #     print(f"Prefix {prefix_len}: {self.prefix_counts[prefix_len]} regular + "
            #         f"{self.sliding_window_counts[prefix_len]} sliding = "
            #         f"{self.total_samples_per_prefix[prefix_len]} total")
                
                # If sequence is long and sliding window is enabled, create multiple windows
                if self.sliding_window and len(input_ids_full) > self.max_length:
                    for j in range(0, len(input_ids_full) - self.max_length + 1, self.window_stride):
                        window_ids = input_ids_full[j:j+self.max_length]
                        padded = self.tokenizer.pad(
                            {'input_ids': window_ids},
                            padding='max_length', 
                            max_length=self.max_length
                        )
                        
                        # Create attention mask
                        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 
                                          for token_id in padded['input_ids']]
                        
                        self.samples.append({
                            'input_ids': padded['input_ids'],
                            'attention_mask': attention_mask,
                            'time_diffs': padded_time_diffs,
                            'next_activity': next_activity,
                            'outcome': outcome,
                            'prefix_length': len(prefix),
                            'customer_type_id': customer_type_id
                        })
                else:
                    # Otherwise, use standard truncation and padding
                    encoding = self.tokenizer(
                        prefix_text,
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation='longest_first',
                        padding='max_length'
                    )
                    
                    self.samples.append({
                        'input_ids': encoding['input_ids'],
                        'attention_mask': encoding['attention_mask'],
                        'time_diffs': padded_time_diffs,
                        'next_activity': next_activity,
                        'outcome': outcome,
                        'prefix_length': len(prefix),
                        'customer_type_id': customer_type_id
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': torch.tensor(sample['input_ids']),
            'attention_mask': torch.tensor(sample['attention_mask']),
            'time_diffs': torch.tensor(sample['time_diffs'], dtype=torch.float),
            'next_activity': torch.tensor(sample['next_activity']).long(),
            'outcome': torch.tensor(sample['outcome']).long(),
            'prefix_length': torch.tensor(sample['prefix_length']).long(),
            'customer_type_id': torch.tensor(sample['customer_type_id']).long()
        }
    

class TaskAwarePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activity_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.outcome_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activity_activation = nn.Tanh()
        self.outcome_activation = nn.Tanh()
        
    def forward(self, hidden_states):
        # Get the first token ([CLS]) representation
        first_token_tensor = hidden_states[:, 0]
        
        # Task-specific pooled outputs
        activity_pooled = self.activity_activation(self.activity_dense(first_token_tensor))
        outcome_pooled = self.outcome_activation(self.outcome_dense(first_token_tensor))
        
        return activity_pooled, outcome_pooled
    

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100, scale_factor=1.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = float(gamma)  # Focusing parameter
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor of shape [N, C] where C is the number of classes
            targets: Tensor of shape [N] with class indices
        """
        # Get log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Get the probability of the target class
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1))
        target_probs = torch.exp(target_log_probs)
        
        # Calculate focal term: (1 - p_t)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Apply class-specific weights
                alpha_t = self.alpha.gather(0, targets)
                focal_weight = focal_weight * alpha_t.unsqueeze(1)
        
        # Compute the focal loss
        loss = -focal_weight * target_log_probs * self.scale_factor

        # Dynamic scaling based on prediction confidence
        confidence_scale = 1.0 / (target_probs.mean() + 1e-6)  # Scale up more when predictions are too confident
        loss = loss * confidence_scale.detach()  # Detach to prevent gradient computation
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            loss = loss[mask]
            if loss.size(0) == 0:
                return torch.tensor(0.0, device=inputs.device)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class DynamicWeightedLoss(nn.Module):
    def __init__(self, initial_activity_weight=0.7, initial_outcome_weight=0.3, 
                 activity_class_weights=None, outcome_class_weights=None,
                 use_focal_loss=False, focal_gamma=2.0):
        super().__init__()
        
        # Store class weights
        self.activity_class_weights = activity_class_weights
        self.outcome_class_weights = outcome_class_weights
        
        # Focal loss flag and parameter
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        # Initialize loss functions based on settings
        if use_focal_loss:
            self.activity_loss_fn = FocalLoss(
                alpha=activity_class_weights, 
                gamma=focal_gamma,
                scale_factor=100.0  # Add scale factor to make losses more meaningful
            )
            self.outcome_loss_fn = FocalLoss(
                alpha=outcome_class_weights, 
                gamma=focal_gamma,
                scale_factor=100.0
            )
            print(f"Using Focal Loss with gamma={focal_gamma}")
        else:
            self.activity_loss_fn = nn.CrossEntropyLoss(weight=activity_class_weights)
            self.outcome_loss_fn = nn.CrossEntropyLoss(weight=outcome_class_weights)
            print("Using Weighted Cross Entropy Loss")
        
        # Initial weights for task balancing
        self.activity_weight = initial_activity_weight
        self.outcome_weight = initial_outcome_weight
        
        # Store initial losses
        self.initial_activity_loss = None
        self.initial_outcome_loss = None
        
        # Parameters for abridged linear schedule
        self.total_steps = 0
        self.current_step = 0
        self.threshold_step = 0  # Will be set once total_steps is known
        
        # Parameter for LBTW
        self.alpha = 0.5
        
        # Track loss history
        self.activity_loss_history = []
        self.outcome_loss_history = []
        self.outcome_target_reached = False
        self.outcome_target_value = 0.01  # Set this to your desired target loss
        
    def set_total_steps(self, total_steps):
        self.total_steps = total_steps
        self.threshold_step = total_steps // 10  # 10% of total steps
    
    def forward(self, activity_logits, outcome_logits, activity_labels, outcome_labels):
        # # Compute individual losses
        # loss_activity = self.activity_loss_fn(activity_logits, activity_labels)
        # loss_outcome = self.outcome_loss_fn(outcome_logits, outcome_labels)
        
        # Initialize losses
        loss_activity = torch.tensor(0.0, device=activity_labels.device if activity_labels is not None else outcome_labels.device)
        loss_outcome = torch.tensor(0.0, device=outcome_labels.device if outcome_labels is not None else activity_labels.device)
        
        # Compute losses only for active heads
        if activity_logits is not None and activity_labels is not None:
            loss_activity = self.activity_loss_fn(activity_logits, activity_labels)
            
        if outcome_logits is not None and outcome_labels is not None:
            loss_outcome = self.outcome_loss_fn(outcome_logits, outcome_labels)
        
        # If only one head is active, return its loss directly
        if activity_logits is None:
            return loss_outcome, loss_activity, loss_outcome
        elif outcome_logits is None:
            return loss_activity, loss_activity, loss_outcome

        # Store initial losses on first forward pass
        if self.initial_activity_loss is None:
            self.initial_activity_loss = loss_activity.item()
            self.initial_outcome_loss = loss_outcome.item()
        
        # Update loss history
        self.activity_loss_history.append(loss_activity.item())
        self.outcome_loss_history.append(loss_outcome.item())
        
        # Check if outcome loss target reached
        if len(self.outcome_loss_history) > 50:  # Need some history to smooth
            avg_outcome_loss = sum(self.outcome_loss_history[-50:]) / 50
            if avg_outcome_loss < self.outcome_target_value and not self.outcome_target_reached:
                print(f"Outcome loss target reached ({avg_outcome_loss:.4f}). Focusing on activity loss.")
                self.outcome_target_reached = True
        
        # Dynamic weighting based on training progress
        if self.outcome_target_reached:
            # Once outcome loss is good, focus more on activity loss
            self.activity_weight = min(0.9, self.activity_weight + 0.01)  # Gradually increase activity weight
            self.outcome_weight = 1.0 - self.activity_weight  # Ensure weights sum to 1
        else:
            # Before outcome target is reached, use the hybrid approach
            # Abridged Linear for activity (main task)
            if self.current_step < self.threshold_step:
                activity_factor = 1.0 - (self.current_step / self.threshold_step) * 0.3
            else:
                activity_factor = 0.7
            
            # LBTW for outcome (auxiliary task)
            if len(self.outcome_loss_history) > 0:
                current_outcome_loss = loss_outcome.item()
                outcome_loss_ratio = current_outcome_loss / self.initial_outcome_loss
                outcome_factor = pow(outcome_loss_ratio, self.alpha)
            else:
                outcome_factor = self.outcome_weight
            
            # Normalize weights to sum to 1
            total = activity_factor + outcome_factor
            self.activity_weight = activity_factor / total
            self.outcome_weight = outcome_factor / total
        
        # Increment step counter
        self.current_step += 1
        
        # Compute weighted loss
        total_loss = self.activity_weight * loss_activity + self.outcome_weight * loss_outcome
        
        return total_loss, loss_activity, loss_outcome


def compute_class_weights(labels, normalize=True):
    """
    Compute class weights with proper safeguards against extreme values.
    """
    # Convert to proper format for bincount
    if isinstance(labels, torch.Tensor):
        labels = labels.long()
    else:
        labels = torch.tensor(labels, dtype=torch.long)
    
    # Handle empty tensor case
    if labels.numel() == 0:
        return torch.tensor([]), torch.tensor([])
    
    # Count occurrences of each class
    class_counts = torch.bincount(labels)
    
    # Add small epsilon to avoid division by zero
    weights = 1.0 / (class_counts.float() + 1e-8)
    
    # Clip weights to reasonable range to avoid extreme values
    weights = torch.clamp(weights, 0.1, 10.0)
    
    # Normalize weights if requested
    if normalize:
        weights = weights / weights.sum()
        
    return weights, class_counts


# ------------------------------
# Dual Output Model for Next Activity and Outcome Prediction
# ------------------------------
class ModernBertDualOutput(ModernBertPreTrainedModel):
    def __init__(self, config, num_activities, num_outcomes, num_customer_types):
        super().__init__(config)
        self.modernbert = ModernBertModel(config)
        
        # Keep track of training mode for embedding access
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Layer normalization for better stability
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Multiple dropout rates for regularization
        self.dropout = nn.Dropout(0.5)
        self.dropout_high = nn.Dropout(0.7)
        self.dropout_low = nn.Dropout(0.4)
        
        hidden_size = config.hidden_size
        
        # Shared processing layer
        self.shared_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Time embeddings with gating
        self.time_embedding = TimeEmbedding(hidden_size)
        self.time_gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Task-specific layers with reduced complexity
        self.activity_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.outcome_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Classification heads with L2 regularization
        self.activity_head = nn.Linear(hidden_size, num_activities)
        self.outcome_head = nn.Linear(hidden_size, num_outcomes)
        self.activity_head.weight.register_hook(lambda grad: grad + 0.01 * self.activity_head.weight)
        self.outcome_head.weight.register_hook(lambda grad: grad + 0.01 * self.outcome_head.weight)
        
        # Task-aware components
        self.task_pooler = TaskAwarePooler(config)
        
        # Customer type specific prompts
        self.prompt_manager = CustomerTypePromptManager(
            hidden_size=config.hidden_size,
            num_customer_types=num_customer_types
        )
        
        # Prompt attention mechanisms
        self.activity_prompt_attention = nn.MultiheadAttention(
            config.hidden_size, 
            num_heads=32,  # Reduced from 8
            dropout=0.1,
            batch_first=True
        )
        self.outcome_prompt_attention = nn.MultiheadAttention(
            config.hidden_size,
            num_heads=8,  # Reduced from 8
            dropout=0.1,
            batch_first=True
        )
        
        self.post_init()

    def save_prompts(self, filepath):
        """Save prompt states to file"""
        prompt_state = {
            'activity_prompts': self.prompt_manager.activity_prompts.data,  # Use .data instead of state_dict()
            'outcome_prompts': self.prompt_manager.outcome_prompts.data,
            'customer_type_metrics': self.prompt_manager.customer_type_metrics,
            'prompt_history': self.prompt_manager.prompt_history
        }
        torch.save(prompt_state, filepath)

    def load_prompts(self, filepath):
        """Load prompt states from file"""
        if not os.path.exists(filepath):
            print(f"No prompt file found at {filepath}")
            return
            
        prompt_state = torch.load(filepath, map_location=self.device)
        
        # Load the prompt tensors directly
        self.prompt_manager.activity_prompts.data.copy_(prompt_state['activity_prompts'])
        self.prompt_manager.outcome_prompts.data.copy_(prompt_state['outcome_prompts'])
        
        # Load the metrics and history
        self.prompt_manager.customer_type_metrics = prompt_state['customer_type_metrics']
        if 'prompt_history' in prompt_state:
            self.prompt_manager.prompt_history = prompt_state['prompt_history']

    def get_input_embeddings(self):
        """Get input embeddings from the underlying BERT model"""
        return self.modernbert.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        """Set input embeddings for the underlying BERT model"""
        self.modernbert.embeddings.tok_embeddings = value

    def forward(self, input_ids, attention_mask=None, time_diffs=None, customer_type_ids=None):
        # Ensure all inputs are on the correct device
        device = input_ids.device
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if time_diffs is not None:
            time_diffs = time_diffs.to(device)
        if customer_type_ids is not None:
            customer_type_ids = customer_type_ids.to(device)
            
        # Get base representations from BERT
        outputs = self.modernbert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply shared processing
        hidden_states = self.shared_hidden(hidden_states)
        
        # Add time embeddings if provided
        if time_diffs is not None:
            time_embeddings = self.time_embedding(time_diffs)
            time_gate = self.time_gate(hidden_states)
            hidden_states = hidden_states + time_gate * time_embeddings

        # Apply prompt attention if enabled and customer types provided
        if hasattr(self, 'use_prompts') and self.use_prompts and customer_type_ids is not None:
            activity_prompts, outcome_prompts = self.prompt_manager.get_prompts(customer_type_ids)

            # Only apply relevant prompts based on active heads
            prompt_context = torch.zeros_like(hidden_states)  # Initialize with zeros
            if hasattr(self, 'use_activity_head') and self.use_activity_head:
                activity_context, _ = self.activity_prompt_attention(
                    hidden_states, activity_prompts, activity_prompts
                )
                prompt_context += activity_context
                
            if hasattr(self, 'use_outcome_head') and self.use_outcome_head:
                outcome_context, _ = self.outcome_prompt_attention(
                    hidden_states, outcome_prompts, outcome_prompts
                )
                prompt_context += outcome_context
                
            # Apply prompt context if any heads are active
            if torch.any(prompt_context != 0):  # Check if any element is non-zero
                prompt_scale = 0.3
                hidden_states = self.layer_norm(
                    hidden_states + 
                    prompt_scale * self.dropout_low(prompt_context)
                )

        # Get task-specific representations only for active heads
        activity_logits = None
        outcome_logits = None
        
        if hasattr(self, 'use_activity_head') and self.use_activity_head:
            activity_pooled, _ = self.task_pooler(hidden_states)
            if activity_pooled.size(0) > 1:
                activity_pooled = self.batch_norm(activity_pooled)
            activity_pooled = self.layer_norm(activity_pooled)
            if self.training:
                activity_pooled = self.dropout_high(activity_pooled)
            activity_hidden = self.activity_hidden(activity_pooled)
            if self.training:
                activity_hidden = self.dropout(activity_hidden)
            activity_logits = self.activity_head(activity_hidden)
        
        if hasattr(self, 'use_outcome_head') and self.use_outcome_head:
            _, outcome_pooled = self.task_pooler(hidden_states)
            if outcome_pooled.size(0) > 1:
                outcome_pooled = self.batch_norm(outcome_pooled)
            outcome_pooled = self.layer_norm(outcome_pooled)
            if self.training:
                outcome_pooled = self.dropout_high(outcome_pooled)
            outcome_hidden = self.outcome_hidden(outcome_pooled)
            if self.training:
                outcome_hidden = self.dropout(outcome_hidden)
            outcome_logits = self.outcome_head(outcome_hidden)
        
        return activity_logits, outcome_logits
            
        #     # Apply attention mechanisms
        #     activity_context, _ = self.activity_prompt_attention(
        #         hidden_states, activity_prompts, activity_prompts
        #     )
        #     outcome_context, _ = self.outcome_prompt_attention(
        #         hidden_states, outcome_prompts, outcome_prompts
        #     )
            
        #     # Controlled residual connections with scaling
        #     prompt_scale = 0.5
        #     hidden_states = self.layer_norm(
        #         hidden_states + 
        #         prompt_scale * self.dropout_low(activity_context + outcome_context)
        #     )

        # # Get task-specific representations
        # activity_pooled, outcome_pooled = self.task_pooler(hidden_states)
        
        # # Apply normalization and regularization
        # if activity_pooled.size(0) > 1:
        #     activity_pooled = self.batch_norm(activity_pooled)
        #     outcome_pooled = self.batch_norm(outcome_pooled)
        
        # activity_pooled = self.layer_norm(activity_pooled)
        # outcome_pooled = self.layer_norm(outcome_pooled)
        
        # # Apply heavy dropout during training
        # if self.training:
        #     activity_pooled = self.dropout_high(activity_pooled)
        #     outcome_pooled = self.dropout_high(outcome_pooled)
        
        # # Process through task-specific layers
        # activity_hidden = self.activity_hidden(activity_pooled)
        # outcome_hidden = self.outcome_hidden(outcome_pooled)
        
        # # Final dropout before classification
        # if self.training:
        #     activity_hidden = self.dropout(activity_hidden)
        #     outcome_hidden = self.dropout(outcome_hidden)
        
        # # Get final predictions
        # activity_logits = self.activity_head(activity_hidden)
        # outcome_logits = self.outcome_head(outcome_hidden)
        
        # return activity_logits, outcome_logits

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.time_projection = nn.Linear(1, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, time_diffs):
        # Reshape time_diffs to (batch_size, sequence_length, 1)
        time_diffs = time_diffs.unsqueeze(-1)
        # Project to hidden_size dimension
        time_embeddings = self.time_projection(time_diffs)
        # Apply layer normalization
        time_embeddings = self.layer_norm(time_embeddings)
        return time_embeddings

class CustomerTypePromptManager:
    def __init__(self, hidden_size, num_customer_types, prompt_length=5):
        self.hidden_size = hidden_size
        self.num_customer_types = num_customer_types
        self.prompt_length = prompt_length
        
        # Initialize activity and outcome prompts for each customer type
        self.activity_prompts = nn.Parameter(
            torch.randn(num_customer_types, prompt_length, hidden_size)
        )
        self.outcome_prompts = nn.Parameter(
            torch.randn(num_customer_types, prompt_length, hidden_size)
        )
        
        # Track performance metrics per customer type - change defaultdict to regular dict
        self.customer_type_metrics = {}
        for ctype in range(num_customer_types):
            self.customer_type_metrics[ctype] = {
                'accuracy': [],
                'predictions': [],
                'actual': []
            }
        
        # Initialize prompt history - change defaultdict to regular dict
        self.prompt_history = {}
        for ctype in range(num_customer_types):
            self.prompt_history[ctype] = {
                'activity_prompts': [],
                'outcome_prompts': [],
                'epochs': []
            }
        
        # Learning rate for prompt updates
        self.prompt_lr = 0.01
        
    def get_prompts(self, customer_type_ids):
        """Get activity and outcome prompts for given customer types"""
        # Get device from prompts
        device = self.activity_prompts.device
        
        # Move customer_type_ids to same device if needed
        if customer_type_ids.device != device:
            customer_type_ids = customer_type_ids.to(device)
            
        batch_activity_prompts = self.activity_prompts[customer_type_ids]
        batch_outcome_prompts = self.outcome_prompts[customer_type_ids]
        
        return batch_activity_prompts, batch_outcome_prompts

    def to(self, device):
        """Move prompts to specified device"""
        self.activity_prompts = nn.Parameter(self.activity_prompts.to(device))
        self.outcome_prompts = nn.Parameter(self.outcome_prompts.to(device))
        return self
    
    def update_prompts(self, customer_type_id, activity_loss, outcome_loss):
        """Update prompts based on prediction performance"""
        try:
            # Only compute gradients if the losses are connected to the graph
            if activity_loss.requires_grad:
                activity_grad = torch.autograd.grad(
                    activity_loss, 
                    self.activity_prompts[customer_type_id],
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if activity_grad is not None:
                    self.activity_prompts.data[customer_type_id] -= self.prompt_lr * activity_grad
            
            if outcome_loss.requires_grad:
                outcome_grad = torch.autograd.grad(
                    outcome_loss, 
                    self.outcome_prompts[customer_type_id],
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if outcome_grad is not None:
                    self.outcome_prompts.data[customer_type_id] -= self.prompt_lr * outcome_grad
                    
        except RuntimeError as e:
            # If there's an error, just skip this update
            print(f"Warning: Could not update prompts for customer type {customer_type_id}")
            pass

    def log_prompt_state(self, customer_type_id, epoch):
        """Log current prompt state for visualization"""
        device = self.activity_prompts.device  # Get the current device
        
        self.prompt_history[customer_type_id]['activity_prompts'].append(
            self.activity_prompts[customer_type_id].detach().clone()  # Keep on same device
        )
        self.prompt_history[customer_type_id]['outcome_prompts'].append(
            self.outcome_prompts[customer_type_id].detach().clone()  # Keep on same device
        )
        self.prompt_history[customer_type_id]['epochs'].append(epoch)

    def get_prompt_statistics(self):
        """Get statistics about prompt changes"""
        stats = {}
        device = self.activity_prompts.device  # Get the current device
        
        for ctype in range(self.num_customer_types):
            if len(self.prompt_history[ctype]['activity_prompts']) > 1:
                # Move tensors to the same device before computing changes
                activity_prompts = [p.to(device) for p in self.prompt_history[ctype]['activity_prompts']]
                outcome_prompts = [p.to(device) for p in self.prompt_history[ctype]['outcome_prompts']]
                
                # Calculate prompt changes over time
                activity_changes = torch.stack([
                    torch.norm(p2 - p1) 
                    for p1, p2 in zip(activity_prompts[:-1], activity_prompts[1:])
                ])
                
                outcome_changes = torch.stack([
                    torch.norm(p2 - p1) 
                    for p1, p2 in zip(outcome_prompts[:-1], outcome_prompts[1:])
                ])
                
                # Move results to CPU for storage in dictionary
                stats[ctype] = {
                    'avg_activity_change': activity_changes.mean().cpu().item(),
                    'max_activity_change': activity_changes.max().cpu().item(),
                    'avg_outcome_change': outcome_changes.mean().cpu().item(),
                    'max_outcome_change': outcome_changes.max().cpu().item()
                }
    
        return stats
# ------------------------------
# Define Weighted Loss Function
# ------------------------------
class WeightedLoss(nn.Module):
    def __init__(self, activity_weight=0.7, outcome_weight=0.3):
        super().__init__()
        self.activity_weight = activity_weight
        self.outcome_weight = outcome_weight
        self.activity_loss = nn.CrossEntropyLoss()
        self.outcome_loss = nn.CrossEntropyLoss()
    
    def forward(self, activity_logits, outcome_logits, activity_labels, outcome_labels):
        loss_activity = self.activity_loss(activity_logits, activity_labels)
        loss_outcome = self.outcome_loss(outcome_logits, outcome_labels)
        total_loss = self.activity_weight * loss_activity + self.outcome_weight * loss_outcome
        return total_loss, loss_activity, loss_outcome

# ------------------------------
# Evaluation Functions
# ------------------------------
# def evaluate_model(model, dataloader, device, task='both', output_file=None):
#     if len(dataloader.dataset) == 0:
#         print("Warning: Empty evaluation dataset!")
#         return None
        
#     print(f"\nEvaluating {len(dataloader.dataset)} samples...")

#     model.eval()
#     all_prefix_lengths = []
#     # For activity prediction
#     activity_true = []
#     activity_pred = []
#     activity_prob = []
#     # For outcome prediction
#     outcome_true = []
#     outcome_pred = []
#     outcome_prob = []
    
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating"):
#             # Add batch size check
#             if len(batch['input_ids']) == 0:
#                 continue
#             inputs = batch['input_ids'].to(device)
#             masks = batch['attention_mask'].to(device)
#             act_labels = batch['next_activity'].to(device)
#             out_labels = batch['outcome'].to(device)
#             prefix_lengths = batch['prefix_length'].cpu().numpy()
#             # Forward pass
#             activity_logits, outcome_logits = model(inputs, attention_mask=masks)
#             # Activity predictions
#             act_probs = torch.softmax(activity_logits, dim=1)
#             act_preds = torch.argmax(activity_logits, dim=1)
#             # Outcome predictions
#             out_probs = torch.softmax(outcome_logits, dim=1)
#             out_preds = torch.argmax(outcome_logits, dim=1)
#             # Store results
#             all_prefix_lengths.extend(prefix_lengths.tolist())
#             activity_true.extend(act_labels.cpu().numpy())
#             activity_pred.extend(act_preds.cpu().numpy())
#             activity_prob.extend(act_probs.cpu().numpy())
#             outcome_true.extend(out_labels.cpu().numpy())
#             outcome_pred.extend(out_preds.cpu().numpy())
#             outcome_prob.extend(out_probs.cpu().numpy())
    
#     results_df = pd.DataFrame({
#         'prefix_length': all_prefix_lengths,
#         'activity_true': activity_true,
#         'activity_pred': activity_pred,
#         'outcome_true': outcome_true,
#         'outcome_pred': outcome_pred
#     })
    
#     if task in ['activity', 'both']:
#         save_task_metrics(
#             results_df,
#             'activity_true',
#             'activity_pred',
#             activity_prob,
#             f"activity_prediction_metrics.txt" if output_file is None else f"{output_file}_activity.txt"
#         )
    
#     if task in ['outcome', 'both']:
#         save_task_metrics(
#             results_df,
#             'outcome_true',
#             'outcome_pred',
#             outcome_prob,
#             f"outcome_prediction_metrics.txt" if output_file is None else f"{output_file}_outcome.txt"
#         )
#     return results_df

def get_next_counter(log_dir):
    """Find the next available counter for file naming"""
    counter = 1
    while True:
        # Check if any of the files with current counter exist
        files_exist = any(os.path.exists(os.path.join(log_dir, f"{base}_{counter}.{ext}"))
                         for base, ext in [("predictions", "csv"),
                                         ("activity_metrics", "txt"),
                                         ("outcome_metrics", "txt"),
                                         ("time_metrics", "txt"),
                                         ("activity_class_distribution", "csv"),
                                         ("outcome_class_distribution", "csv")])
        if not files_exist:
            return counter
        counter += 1

def evaluate_model(model, dataloader, device, log, output_dir, use_focal_loss, use_class_weights, 
                  use_prompts, use_prompt_updates, focal_gamma, mam_flag,
                  use_activity_head, use_outcome_head, task='both'):
    """Evaluate model and save predictions in standardized format"""
    
    if len(dataloader.dataset) == 0:
        print("Warning: Empty evaluation dataset!")
        return None
        
    print(f"\nEvaluating {len(dataloader.dataset)} samples...")
    eval_start_time = time()

    model.eval()
    predictions = {
        'prefix_length': [],
        'activity_true': [],
        'activity_pred': [],
        'outcome_true': [],
        'outcome_pred': []
    }
    
    # Initialize probability lists only for active heads
    activity_probs = [] if use_activity_head else None
    outcome_probs = [] if use_outcome_head else None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch['input_ids']) == 0:
                continue
                
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            act_labels = batch['next_activity'].to(device)
            out_labels = batch['outcome'].to(device)
            prefix_lengths = batch['prefix_length'].cpu().numpy()
            
            # Forward pass
            activity_logits, outcome_logits = model(inputs, attention_mask=masks)
            
            # Process predictions only for active heads
            if use_activity_head:
                act_probs = torch.softmax(activity_logits, dim=1)
                act_preds = torch.argmax(activity_logits, dim=1)
                predictions['activity_true'].extend(act_labels.cpu().numpy())
                predictions['activity_pred'].extend(act_preds.cpu().numpy())
                activity_probs.append(act_probs.cpu().numpy())
            else:
                # Fill with placeholder values when head is inactive
                predictions['activity_true'].extend([-1] * len(prefix_lengths))
                predictions['activity_pred'].extend([-1] * len(prefix_lengths))
            
            if use_outcome_head:
                out_probs = torch.softmax(outcome_logits, dim=1)
                out_preds = torch.argmax(outcome_logits, dim=1)
                predictions['outcome_true'].extend(out_labels.cpu().numpy())
                predictions['outcome_pred'].extend(out_preds.cpu().numpy())
                outcome_probs.append(out_probs.cpu().numpy())
            else:
                # Fill with placeholder values when head is inactive
                predictions['outcome_true'].extend([-1] * len(prefix_lengths))
                predictions['outcome_pred'].extend([-1] * len(prefix_lengths))
            
            predictions['prefix_length'].extend(prefix_lengths.tolist())
    
    # Calculate total evaluation time
    eval_time = time() - eval_start_time
    
    # Convert to DataFrame
    results_df = pd.DataFrame(predictions)
    
    # Stack probability arrays only for active heads
    if use_activity_head:
        activity_probs = np.vstack(activity_probs)
    if use_outcome_head:
        outcome_probs = np.vstack(outcome_probs)
    
    # Add probability columns only for active heads
    if use_activity_head:
        for i in range(activity_probs.shape[1]):
            results_df[f'activity_prob_{i}'] = activity_probs[:, i]
            
    if use_outcome_head:
        for i in range(outcome_probs.shape[1]):
            results_df[f'outcome_prob_{i}'] = outcome_probs[:, i]
    
    # Calculate class distributions only for active heads
    prefix_lengths = sorted(results_df['prefix_length'].unique())
    activity_dist = pd.DataFrame(index=prefix_lengths) if use_activity_head else None
    outcome_dist = pd.DataFrame(index=prefix_lengths) if use_outcome_head else None
    
    # Calculate distributions for each prefix length
    for prefix_len in prefix_lengths:
        prefix_mask = results_df['prefix_length'] == prefix_len
        
        if use_activity_head:
            act_counts = results_df[prefix_mask]['activity_true'].value_counts()
            for class_idx in range(activity_probs.shape[1]):
                activity_dist.loc[prefix_len, f'activity_class_{class_idx}'] = act_counts.get(class_idx, 0)
                
        if use_outcome_head:
            out_counts = results_df[prefix_mask]['outcome_true'].value_counts()
            for class_idx in range(outcome_probs.shape[1]):
                outcome_dist.loc[prefix_len, f'outcome_class_{class_idx}'] = out_counts.get(class_idx, 0)
    
    # Save results
    config_str = get_config_string(
        use_focal_loss, use_class_weights, use_prompts, use_prompt_updates,
        focal_gamma, mam_flag, use_activity_head, use_outcome_head
    )
    # Setup output directory
    config_dir = setup_output_directories(output_dir, config_str)
    os.makedirs(config_dir, exist_ok=True)

    # Save predictions
    results_df.to_csv(f"{config_dir}/predictions_{config_str}.csv", index=False)

    # Save class distributions for active heads
    if use_activity_head:
        activity_dist.to_csv(f"{config_dir}/activity_class_distribution_{config_str}.csv")
    if use_outcome_head:
        outcome_dist.to_csv(f"{config_dir}/outcome_class_distribution_{config_str}.csv")

    # Save metrics for active heads
    if task in ['activity', 'both'] and use_activity_head:
        save_task_metrics(
            results_df,
            'activity_true',
            'activity_pred',
            activity_probs,
            f"{config_dir}/activity_metrics_{config_str}.txt"
        )

    if task in ['outcome', 'both'] and use_outcome_head:
        save_task_metrics(
            results_df,
            'outcome_true',
            'outcome_pred',
            outcome_probs,
            f"{config_dir}/outcome_metrics_{config_str}.txt"
        )

    # Save timing metrics
    metrics_file = f"{config_dir}/time_metrics_{config_str}.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation time: {eval_time:.2f} seconds\n")
        f.write(f"Samples evaluated: {len(dataloader.dataset)}\n")
        f.write(f"Average time per sample: {eval_time/len(dataloader.dataset):.4f} seconds\n")
    
    return results_df, activity_dist, outcome_dist, eval_time

def save_task_metrics(results_df, true_col, pred_col, prob_scores, output_file):
    # Overall metrics
    y_true = results_df[true_col].values
    y_pred = results_df[pred_col].values
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f'Overall metrics from {output_file}: ')
    print(f'Accuracy: {accuracy} \nF1: {f1} \nPrecision: {precision} \nRecall: {recall}')
    try:
        roc_auc = roc_auc_score(y_true, prob_scores, multi_class='ovr')
    except:
        roc_auc = float('nan')
    
    # Metrics per prefix length
    prefix_metrics = []
    for length in sorted(results_df['prefix_length'].unique()):
        subset = results_df[results_df['prefix_length'] == length]
        sub_true = subset[true_col].values
        sub_pred = subset[pred_col].values
        # if len(sub_true) < 5:
        #     continue
        sub_acc = accuracy_score(sub_true, sub_pred)
        sub_f1 = f1_score(sub_true, sub_pred, average='weighted', zero_division=0)
        sub_prec = precision_score(sub_true, sub_pred, average='weighted', zero_division=0)
        sub_recall = recall_score(sub_true, sub_pred, average='weighted', zero_division=0)
        try:
            sub_indices = results_df['prefix_length'] == length
            sub_probs = np.array(prob_scores)[sub_indices]
            sub_roc = roc_auc_score(sub_true, sub_probs, multi_class='ovr')
        except:
            sub_roc = float('nan')
        prefix_metrics.append({
            'Length': length,
            'Accuracy': sub_acc,
            'F1': sub_f1,
            'Precision': sub_prec,
            'Recall': sub_recall,
            'ROC_AUC': sub_roc,
            'NumSamples': len(sub_true)
        })
        
    with open(output_file, 'w') as f:
        f.write("Overall Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n\n")
        f.write(f"ROC AUC: {roc_auc if not np.isnan(roc_auc) else 'nan'}\n\n")
        f.write("Metrics per prefix length:\n")
        f.write("Length;Accuracy;F1;Precision;Recall;ROC_AUC;NumSamples\n")
        for m in prefix_metrics:
            f.write(f"{int(m['Length'])};{m['Accuracy']:.4f};{m['F1']:.4f};{m['Precision']:.4f};" +
                    f"{m['Recall']:.4f};{m['ROC_AUC'] if not np.isnan(m['ROC_AUC']) else 'nan'};{m['NumSamples']}\n")
    print(f"Metrics saved to {output_file}")


def customize_tokenizer_for_activities(df, base_tokenizer):
    # Get all unique activities
    unique_activities = df['Activity'].unique()
    
    # Split activities at '_' and collect unique components
    activity_components = set()
    for activity in unique_activities:
        components = activity.split('_')
        for component in components:
            if component:  # Skip empty components
                activity_components.add(component)
    
    # Add components as special tokens
    activity_components = list(activity_components)
    print(f"Adding {len(activity_components)} activity components as special tokens")
    special_tokens_dict = {"additional_special_tokens": activity_components + ["[SEP]"]}
    base_tokenizer.add_special_tokens(special_tokens_dict)
    
    return base_tokenizer


def calculate_sequence_stats(trace_ids, df, tokenizer):
    lengths = []
    for trace_id in trace_ids:
        group = df[df['trace_id'] == trace_id]
        # Sort activities by timestamp and join them together into a single text
        activities = group.sort_values('TimestampContact')['Activity'].tolist()
        text = ' '.join(activities)
        encoding = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(encoding))
    return np.max(lengths), np.median(lengths), np.min(lengths)

def calculate_input_sequence_stats(trace_ids, df):
    lengths = []
    for trace_id in trace_ids:
        group = df[df['trace_id'] == trace_id]
        activities = group.sort_values('TimestampContact')['Activity'].tolist()
        lengths.append(len(activities))
    return np.median(lengths), np.mean(lengths), np.std(lengths)

def verify_dataset_alignment(dataset, num_activities, num_outcomes):
    """Verify dataset alignment and distribution"""
    activity_counts = defaultdict(int)
    outcome_counts = defaultdict(int)
    
    for sample in dataset.samples:
        activity_counts[sample['next_activity']] += 1
        outcome_counts[sample['outcome']] += 1
        
        # Verify sequence lengths match
        assert len(sample['input_ids']) == len(sample['attention_mask']), \
            f"Misaligned sequence lengths: {len(sample['input_ids'])} vs {len(sample['attention_mask'])}"
        
        # Verify labels are within range
        assert 0 <= sample['next_activity'] < num_activities, \
            f"Invalid activity label: {sample['next_activity']}"
        assert 0 <= sample['outcome'] < num_outcomes, \
            f"Invalid outcome label: {sample['outcome']}"
    
    # print("\nActivity distribution:")
    # for act, count in activity_counts.items():
    #     print(f"Activity {act}: {count} samples ({count/len(dataset)*100:.2f}%)")
    
    # print("\nOutcome distribution:")
    # for out, count in outcome_counts.items():
    #     print(f"Outcome {out}: {count} samples ({count/len(dataset)*100:.2f}%)")


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed
        return 0
    except:
        return 0

# Add this function for more detailed memory tracking
def get_detailed_gpu_stats():
    """Get detailed GPU memory statistics from PyTorch"""
    if torch.cuda.is_available():
        stats = memory_stats()
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2  # MB
        }
    return {'allocated': 0, 'cached': 0, 'max_allocated': 0}

def get_config_string(use_focal_loss, use_class_weights, use_prompts, use_prompt_updates, 
                     focal_gamma, mam_flag, use_activity_head, use_outcome_head):
    """Generate a concise configuration string for file naming"""
    config = []
    
    # Add model configuration indicators
    if use_activity_head and use_outcome_head:
        config.append("dual")
    elif use_activity_head:
        config.append("act")
    elif use_outcome_head:
        config.append("out")
        
    # Add loss configuration
    if use_focal_loss:
        config.append(f"fl{focal_gamma}")
    if use_class_weights:
        config.append("cw")
        
    # Add prompt configuration
    if use_prompts:
        prompt_config = "p"
        if use_prompt_updates:
            prompt_config += "u"
        config.append(prompt_config)
        
    # Add MAM flag
    if mam_flag:
        config.append("mam")
    
    return "_".join(config)

def setup_output_directories(base_output_dir, config_str):
    """Create and return output directory structure"""
    config_dir = os.path.join(base_output_dir, config_str)
    os.makedirs(config_dir, exist_ok=True)
    return config_dir

def decode_activities(input_ids, tokenizer, max_tokens=10):
    """Decode and format the first few tokens of input sequences"""
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    tokens = decoded.split()
    return ' '.join(tokens[:max_tokens]) + ('...' if len(tokens) > max_tokens else '')


# ------------------------------
# Training and Validation
# ------------------------------
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='Base output directory')
    parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
    parser.add_argument('--use_prompts', type=str, default='true', help='Whether to use prompts')
    parser.add_argument('--use_prompt_updates', type=str, default='true', help='Whether to update prompts')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    parser.add_argument('--mam_flag', type=str, default='true', help='Whether to use MAM')
    parser.add_argument('--use_activity_head', type=str, default='true', help='Whether to use activity head')
    parser.add_argument('--use_outcome_head', type=str, default='true', help='Whether to use outcome head')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize timing variables
    start_time = time()
    training_start_time = None
    training_end_time = None
    total_training_time = 0
    total_eval_time = 0
    
    # Configuration for loss function
    use_focal_loss = True  # Set to True to use Focal Loss instead of weighted CE
    focal_gamma = 2.0       # Gamma parameter for focal loss
    use_class_weights = True  # Set to False to disable class weighting
    use_prompts = True  # New flag for ablation study
    use_prompt_updates = True  # New flag for ablation study
    mam_flag = True
    use_activity_head = True  # Default to using activity head
    use_outcome_head = True   # Default to using outcome head

    # Convert string arguments to boolean
    use_prompts = args.use_prompts.lower() == 'true'
    use_prompt_updates = args.use_prompt_updates.lower() == 'true'
    focal_gamma = args.focal_gamma
    mam_flag = args.mam_flag.lower() == 'true'
    use_activity_head = args.use_activity_head.lower() == 'true'
    use_outcome_head = args.use_outcome_head.lower() == 'true'
    log = args.log
    output_dir = args.output_dir

    if not use_activity_head and not use_outcome_head:
        raise ValueError("At least one prediction head (activity or outcome) must be enabled")

    print(f'Currently training on: {log}')
    print(f'Using focal loss: {use_focal_loss}')
    print(f'Using class weights: {use_class_weights}')
    print(f'Using prompts: {use_prompts}')
    print(f'Using prompt updates: {use_prompt_updates}')
    print(f'Focal loss gamma: {focal_gamma}')
    print(f'Using MAM: {mam_flag}')
    print(f'Using activity head: {use_activity_head}')
    print(f'Using outcome head: {use_outcome_head}')

    # Load dataset splits
    trainval_df = pd.read_csv(f"../datasets/{log}/{log}_train-val.csv", encoding='latin-1')
    test_df = pd.read_csv(f"../datasets/{log}/{log}_test.csv", encoding='latin-1')

    def create_development_split(df, max_traces=10):
        """Create a limited but balanced development split"""
        # Get unique traces with their outcomes
        trace_outcomes = df.groupby('trace_id')['outcome'].first()
        
        # Get equal number of traces per outcome
        traces_per_outcome = max(2, max_traces // len(trace_outcomes.unique()))
        selected_traces = []
        
        for outcome in trace_outcomes.unique():
            outcome_traces = trace_outcomes[trace_outcomes == outcome].index
            n_traces = min(len(outcome_traces), traces_per_outcome)
            if n_traces > 0:
                selected = np.random.choice(outcome_traces, n_traces, replace=False)
                selected_traces.extend(selected)
        
        # Create stratification labels only for selected traces
        stratify_labels = df[df['trace_id'].isin(selected_traces)].groupby('trace_id')['outcome'].first().fillna('Transit')
        
        return selected_traces, stratify_labels

    # In main(), modify the dataset creation:
    # Set development mode flag
    dev_mode = False  # Set to False for full training
    max_traces = 50 if dev_mode else None

    if dev_mode:
        print(f"\nRunning in development mode with {max_traces} traces")
        
        # Get limited traces and their stratification labels
        limited_traces, stratify_labels = create_development_split(trainval_df, max_traces)
        # limited_test_traces, _ = create_development_split(test_df, max_traces // 2)
        
        # Split limited traces into train/val
        train_traces, val_traces = train_test_split(
            limited_traces,
            test_size=0.1,
            random_state=42,
            stratify=stratify_labels
        )
        test_traces = test_df['trace_id'].unique()

        print(f"Selected {len(limited_traces)} traces for development")
        print(f"Train traces: {len(train_traces)}")
        print(f"Val traces: {len(val_traces)}")
        print(f"Test traces: {len(test_traces)}")
    else:
        # Split traces into train/val (90/10 split)
        train_traces, val_traces = train_test_split(
            trainval_df['trace_id'].unique(),
            test_size=0.1,
            random_state=42,
            stratify=trainval_df.groupby('trace_id')['outcome'].first().fillna('Transit') # TODO: Check what to do with the missing outcomes, currently replacing w/ transit
        )
        # First select the traces
        # test_size = 100  # Number of traces for testing
        # all_traces = trainval_df['trace_id'].unique()[:test_size]
        
        # # Get stratification labels only for selected traces
        # trace_outcomes = trainval_df[trainval_df['trace_id'].isin(all_traces)].groupby('trace_id')['outcome'].first().fillna('Transit')
        
        # # Now do the split with matching arrays
        # train_traces, val_traces = train_test_split(
        #     all_traces,
        #     test_size=0.1,
        #     random_state=42,
        #     stratify=trace_outcomes
        # )
        test_traces = test_df['trace_id'].unique()

    print("\nDataset sizes:")
    print(f"Train-val set: {len(trainval_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print(f"Number of train traces: {len(train_traces)}")
    print(f"Number of val traces: {len(val_traces)}")
    print(f"Number of test traces: {len(test_traces)}\n")


    # Create label encoders and fit on the entire dataset
    activity_encoder = LabelEncoder()
    outcome_encoder = LabelEncoder()
    customer_type_encoder = LabelEncoder()

    all_activities = pd.concat([trainval_df['Activity'], test_df['Activity']]).unique()
    all_outcomes = pd.concat([trainval_df['outcome'], test_df['outcome']]).unique()
    all_customer_types = pd.concat([trainval_df['type_of_customer'], test_df['type_of_customer']]).unique()
    
    activity_encoder.fit(all_activities)
    outcome_encoder.fit(all_outcomes)
    customer_type_encoder.fit(all_customer_types)
    
    # print("\nUnique labels:")
    # print(f"Activities: {len(activity_encoder.classes_)}")
    # print(f"Outcomes: {len(outcome_encoder.classes_)}")
    # print("\nActivity labels:", activity_encoder.classes_)
    # print("Outcome labels:", outcome_encoder.classes_)
    num_activities = len(activity_encoder.classes_)
    num_outcomes = len(outcome_encoder.classes_)
    num_customer_types = len(customer_type_encoder.classes_)
    # print(f"Number of unique activities: {num_activities}")
    # print(f"Number of unique outcomes: {num_outcomes}")
    
    # Initialize tokenizer with activity components as special tokens
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer = customize_tokenizer_for_activities(trainval_df, tokenizer)

   # Calculate non-tokenized sequence statistics (activity count)
    train_input_max, train_input_median, train_input_min = calculate_input_sequence_stats(train_traces, trainval_df)
    val_input_max, val_input_median, val_input_min = calculate_input_sequence_stats(val_traces, trainval_df)
    test_input_max, test_input_median, test_input_min = calculate_input_sequence_stats(test_traces, test_df)

    # Assume trainval_df and tokenizer have been loaded and preprocessed already
    train_max, train_median, train_min = calculate_sequence_stats(train_traces, trainval_df, tokenizer)
    val_max, val_median, val_min = calculate_sequence_stats(val_traces, trainval_df, tokenizer)
    test_max, test_median, test_min = calculate_sequence_stats(test_traces, test_df, tokenizer)

    print("\nINPUT SEQUENCE LENGTHS (number of activities per trace):")
    print(f"Train set -- Max: {train_input_max}, Median: {train_input_median}, Min: {train_input_min}")
    print(f"Validation set -- Max: {val_input_max}, Median: {val_input_median}, Min: {val_input_min}")
    print(f"Test set -- Max: {test_input_max}, Median: {test_input_median}, Min: {test_input_min}")

    print("\nTOKENIZED SEQUENCE LENGTHS (token count after tokenization):")
    print("Train set sequence lengths -- Max: {}, Median: {}, Min: {}".format(train_max, train_median, train_min))
    print("Validation set sequence lengths -- Max: {}, Median: {}, Min: {}".format(val_max, val_median, val_min))
    print("Test set sequence lengths -- Max: {}, Median: {}, Min: {}".format(test_max, test_median, test_min))

    # Calculate normalization parameters from train-val set
    normalization_params = calculate_normalization_params(trainval_df)
    print("\nTime difference normalization parameters from train-val set:")
    print(f"Min: {normalization_params[0]}, Max: {normalization_params[1]}")

    # Create datasets with encoded labels and consistent normalization
    def create_dataset(trace_ids, df, sliding_window=True):
        df_subset = df[df['trace_id'].isin(trace_ids)].copy()
        dataset = ProcessTraceDataset(
            df_subset, 
            tokenizer, 
            max_length=64,
            sliding_window=sliding_window,
            normalization_params=normalization_params  # Pass normalization parameters
        )
        
        # Encode labels (now working  with samples directly)
        for sample in dataset.samples:
            sample['next_activity'] = activity_encoder.transform([sample['next_activity']])[0]
            sample['outcome'] = outcome_encoder.transform([sample['outcome']])[0]
        
        return dataset

    # Create datasets with consistent normalization
    train_dataset = create_dataset(train_traces, trainval_df, sliding_window=True)
    val_dataset = create_dataset(val_traces, trainval_df, sliding_window=True)
    test_dataset = create_dataset(test_traces, test_df, sliding_window=True)
    
    # Add verification calls
    verify_dataset_alignment(train_dataset, num_activities, num_outcomes)
    verify_dataset_alignment(val_dataset, num_activities, num_outcomes)
    verify_dataset_alignment(test_dataset, num_activities, num_outcomes)
    
    # After creating datasets, add these checks
    print("\nDataset samples:")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples\n")

    # Add prefix analysis prints
    print("\nPrefix Sample Analysis:")
    print("Train dataset prefix counts:", train_dataset.prefix_counts)
    print("Train dataset sliding window counts:", train_dataset.sliding_window_counts)
    
    print("\nValidation dataset prefix counts:", val_dataset.prefix_counts)
    print("Validation dataset sliding window counts:", val_dataset.sliding_window_counts)
    
    print("\nTest dataset prefix counts:", test_dataset.prefix_counts)
    print("Test dataset sliding window counts:", test_dataset.sliding_window_counts)


    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty!")

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # ------------------------------
    # Load Pretrained Transformer if Available
    # ------------------------------
    pretrained_path = os.path.join(output_dir, "mam_pretrained")
    # Check if either file exists together with config.json
    if os.path.exists(pretrained_path) and os.path.exists(os.path.join(pretrained_path, "config.json")) and (
        os.path.exists(os.path.join(pretrained_path, "pytorch_model.bin")) or os.path.exists(os.path.join(pretrained_path, "model.safetensors"))) and mam_flag == True:
        print("Found pretrained MAM model in:", pretrained_path)
        config = ModernBertConfig.from_pretrained(pretrained_path)
    else:
        print("Pretrained model not found, loading base configuration")
        config = ModernBertConfig.from_pretrained("answerdotai/ModernBERT-base")
    
    # Update vocabulary size from tokenizer
    config.vocab_size = len(tokenizer)
    
    # Initialize the dual-output model
    model = ModernBertDualOutput(
        config=config,
        num_activities=num_activities,
        num_outcomes=num_outcomes,
        num_customer_types=num_customer_types
    )
    # Add ablation flags to model
    model.use_prompts = use_prompts
    model.use_prompt_updates = use_prompt_updates

    # After model initialization
    model.use_activity_head = use_activity_head
    model.use_outcome_head = use_outcome_head

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    # Ensure all model components are on the same device
    model.prompt_manager = model.prompt_manager.to(device)
    model.activity_prompt_attention = model.activity_prompt_attention.to(device)
    model.outcome_prompt_attention = model.outcome_prompt_attention.to(device)
    # If a pretrained model is found, load its transformer weights into the current model
    if os.path.exists(pretrained_path) and all(os.path.exists(os.path.join(pretrained_path, f)) for f in ["pytorch_model.bin", "config.json"]):
        pretrained_dict = torch.load(os.path.join(pretrained_path, "pytorch_model.bin"), map_location=device)
        model.modernbert.load_state_dict(pretrained_dict, strict=False)
        print("Loaded pretrained transformer weights from", pretrained_path)
    
    prompt_path = f"{output_dir}/prompt_state.pt"
    if os.path.exists(prompt_path):
        print(f"Loading existing prompts from {prompt_path}")
        model.load_prompts(prompt_path)
    
    # Compute class weights for activities and outcomes
    all_activities = [sample['next_activity'] for sample in train_dataset.samples]
    all_outcomes = [sample['outcome'] for sample in train_dataset.samples]
    
    # Calculate class weights and counts
    activity_class_weights, activity_counts = compute_class_weights(all_activities)
    outcome_class_weights, outcome_counts = compute_class_weights(all_outcomes)
    
    # # Print class weight information
    # print("\nActivity class distribution and weights:")
    # for idx, (activity, weight) in enumerate(zip(activity_encoder.classes_, activity_class_weights)):
    #     count = activity_counts[idx].item()
    #     percentage = 100 * count / sum(activity_counts)
    #     print(f"  {activity}: {weight.item():.4f} (count: {count}, {percentage:.2f}%)")
    
    # print("\nOutcome class distribution and weights:")
    # for idx, (outcome, weight) in enumerate(zip(outcome_encoder.classes_, outcome_class_weights)):
    #     count = outcome_counts[idx].item()
    #     percentage = 100 * count / sum(outcome_counts)
    #     print(f"  {outcome}: {weight.item():.4f} (count: {count}, {percentage:.2f}%)")
    
    # Apply class weights based on configuration
    if not use_class_weights:
        print("Class weighting disabled")
        activity_class_weights = None
        outcome_class_weights = None
    else:
        # Move weights to device
        activity_class_weights = activity_class_weights.to(device)
        outcome_class_weights = outcome_class_weights.to(device)
    
        
    # ------------------------------
    # Training Setup
    # ------------------------------
    # criterion = WeightedLoss(activity_weight=0.7, outcome_weight=0.3)
    # Initialize loss function with class weights and focal loss option
    criterion = DynamicWeightedLoss(
        initial_activity_weight=0.7, 
        initial_outcome_weight=0.3,
        activity_class_weights=activity_class_weights,
        outcome_class_weights=outcome_class_weights,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma
    )    

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # increased from 0.01
    num_epochs = 20  # Increase epochs and use early stopping
    patience = 2
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Initialize tracking variables
    total_gpu_memory = []
    epoch_times = []
    # Calculate total steps and set in criterion
    total_steps = num_epochs * len(train_dataloader)
    criterion.set_total_steps(total_steps)

    # Start training time measurement
    training_start_time = time()

    for epoch in range(num_epochs):
        epoch_start = time()
        gpu_memory_samples = []
        model.train()
        total_train_loss = 0
        total_activity_loss = 0
        total_outcome_loss = 0
        
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch")
        for batch_idx, batch in enumerate(train_iterator):
            batch_start = time()
            if batch_idx % 100 == 0:
                gpu_memory_samples.append(get_gpu_memory_usage())
                gpu_stats = get_detailed_gpu_stats()
                # print(f" GPU Memory: Allocated={gpu_stats['allocated']:.2f}MB, "
                #     f"Cached={gpu_stats['cached']:.2f}MB, "
                #     f"Peak={gpu_stats['max_allocated']:.2f}MB")
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            time_diffs = batch['time_diffs'].to(device)
            activity_labels = batch['next_activity'].to(device)
            outcome_labels = batch['outcome'].to(device)
            customer_type_ids = batch['customer_type_id'].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            activity_logits, outcome_logits = model(
                inputs, 
                attention_mask=masks,
                time_diffs=time_diffs,
                customer_type_ids=customer_type_ids
            )
            # Calculate losses
            loss_total, loss_activity, loss_outcome = criterion(
                activity_logits, 
                outcome_logits, 
                activity_labels, 
                outcome_labels
            )
            # Calculate per-sample losses for prompt updates
            per_sample_act_losses, per_sample_out_losses = compute_per_sample_losses(
                activity_logits, 
                outcome_logits, 
                activity_labels, 
                outcome_labels
            )
            loss_total.backward(retain_graph=True)
            # Update prompts only if both flags are enabled
            if model.use_prompts and model.use_prompt_updates:
                for i, ctype in enumerate(customer_type_ids):
                    model.prompt_manager.update_prompts(
                        ctype.item(),
                        per_sample_act_losses[i],
                        per_sample_out_losses[i]
                    )
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_train_loss += loss_total.item()
            total_activity_loss += loss_activity.item()
            total_outcome_loss += loss_outcome.item()
            batch_time = time() - batch_start
            train_iterator.set_postfix({
                'Total Loss': f"{loss_total.item():.4f}",
                'Activity Loss': f"{loss_activity.item():.4f}",
                'Outcome Loss': f"{loss_outcome.item():.4f}",
                'Batch Time': f"{batch_time:.2f}s",
                'Act Weight': f"{criterion.activity_weight:.2f}",
                'Out Weight': f"{criterion.outcome_weight:.2f}"
            })
        
        # After epoch completion
        epoch_time = time() - epoch_start
        epoch_times.append(epoch_time)
        # Calculate average GPU memory for this epoch
        avg_gpu_memory = sum(gpu_memory_samples) / len(gpu_memory_samples) if gpu_memory_samples else 0
        total_gpu_memory.append(avg_gpu_memory)    

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_activity_loss = 0
        val_outcome_loss = 0

        low_loss_samples_found = 0
        max_low_loss_samples = 10

        # Find the validation loop section and replace with:

        val_iterator = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit="batch")
        with torch.no_grad():
            for batch in val_iterator:
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                activity_labels = batch['next_activity'].to(device)
                outcome_labels = batch['outcome'].to(device)
                activity_logits, outcome_logits = model(inputs, attention_mask=masks)
                loss_total, loss_activity, loss_outcome = criterion(activity_logits, outcome_logits, activity_labels, outcome_labels)
                total_val_loss += loss_total.item()
                val_activity_loss += loss_activity.item()
                val_outcome_loss += loss_outcome.item()
                val_iterator.set_postfix({
                    'Val Total Loss': f"{loss_total.item():.4f}",
                    'Val Act Loss': f"{loss_activity.item():.4f}",
                    'Val Out Loss': f"{loss_outcome.item():.4f}"
                })
                
                # Changed total_loss to loss_total to match the variable name
                # if loss_total.item() < 1.0 and low_loss_samples_found < max_low_loss_samples:
                #     print(f"\nLOW LOSS SAMPLE (#{low_loss_samples_found+1}):")
                #     print(f"Val Total Loss={loss_total.item():.4f}, Val Act Loss={loss_activity.item():.4f}, Val Out Loss={loss_outcome.item():.4f}")
                    
                #     # Optional: print more details about the batch
                #     print(f"Activity logits shape: {activity_logits.shape}")
                #     print(f"Outcome logits shape: {outcome_logits.shape}")
                #     print(f"First sample outcome label: {outcome_labels[0].item()}")
                #     low_loss_samples_found += 1

                # if loss_activity.item() > 10.0 or loss_outcome.item() > 10.0:
                #     print(f"\n{'='*50}")
                #     print(f"HIGH LOSS SAMPLE DETECTED:")
                #     print(f"Val Total Loss={loss_total.item():.4f}, Val Act Loss={loss_activity.item():.4f}, Val Out Loss={loss_outcome.item():.4f}")
                    
                    # batch_size = len(inputs)
                    # for i in range(batch_size):
                    #     print(f"\nSample {i+1}/{batch_size}:")
                    #     # Decode and print input sequence
                    #     print(f"Input sequence: {decode_activities(inputs[i], tokenizer)}")
                        
                    #     # Print activity information if activity loss is high
                    #     if loss_activity.item() > 10.0:
                    #         pred_act = torch.argmax(activity_logits[i]).item()
                    #         true_act = activity_labels[i].item()
                    #         act_probs = F.softmax(activity_logits[i], dim=0)
                    #         print(f"Activity prediction:")
                    #         print(f"  True: {activity_encoder.inverse_transform([true_act])[0]}")
                    #         print(f"  Predicted: {activity_encoder.inverse_transform([pred_act])[0]}")
                    #         print(f"  Confidence: {act_probs[pred_act]:.4f}")
                        
                    #     # Print outcome information if outcome loss is high
                    #     if loss_outcome.item() > 10.0:
                    #         pred_out = torch.argmax(outcome_logits[i]).item()
                    #         true_out = outcome_labels[i].item()
                    #         out_probs = F.softmax(outcome_logits[i], dim=0)
                    #         print(f"Outcome prediction:")
                    #         print(f"  True: {outcome_encoder.inverse_transform([true_out])[0]}")
                    #         print(f"  Predicted: {outcome_encoder.inverse_transform([pred_out])[0]}")
                    #         print(f"  Confidence: {out_probs[pred_out]:.4f}")
                    
                    # print(f"{'='*50}")
        
        epoch_time = time() - epoch_start
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_act_loss = total_activity_loss / len(train_dataloader)
        avg_out_loss = total_outcome_loss / len(train_dataloader)
       
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_act = val_activity_loss / len(val_dataloader)
        avg_val_out = val_outcome_loss / len(val_dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f" Training => Total: {avg_train_loss:.4f} | Activity: {avg_act_loss:.4f} | Outcome: {avg_out_loss:.4f}")
        print(f" Validation => Total: {avg_val_loss:.4f} | Activity: {avg_val_act:.4f} | Outcome: {avg_val_out:.4f}")
        print(f" Epoch Time: {epoch_time:.2f} seconds")
        print(f" GPU Memory Usage: {avg_gpu_memory:.2f} MB\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Create directory if it doesn't exist
            save_dir = output_dir
            os.makedirs(save_dir, exist_ok=True)
            
            try:
                # First save to a temporary file
                temp_model_path = os.path.join(save_dir, 'temp_model.pt')
                torch.save(model.state_dict(), temp_model_path)
                
                # If successful, rename to final filename
                final_model_path = os.path.join(save_dir, 'best_model.pt')
                os.replace(temp_model_path, final_model_path)
                
                # Save prompts
                model.save_prompts(os.path.join(save_dir, 'prompt_state.pt'))
                
                print(f"Model improved, saved checkpoint (val_loss: {avg_val_loss:.4f})")
                print(f"Saved model and prompts to {save_dir}/")
            except Exception as e:
                print(f"Error saving model: {str(e)}")
                print(f"Directory permissions: ")
                os.system(f"ls -l {save_dir}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # After validation, log prompt states
        for ctype in range(num_customer_types):
            model.prompt_manager.log_prompt_state(ctype, epoch)
        
        # Print prompt statistics every few epochs
        # if (epoch + 1) % 5 == 0:
        #     stats = model.prompt_manager.get_prompt_statistics()
        #     print("\nPrompt Evolution Statistics:")
        #     for ctype, stat in stats.items():
        #         print(f"\nCustomer Type {ctype}:")
        #         print(f"Avg Activity Change: {stat['avg_activity_change']:.4f}")
        #         print(f"Max Activity Change: {stat['max_activity_change']:.4f}")
        #         print(f"Avg Outcome Change: {stat['avg_outcome_change']:.4f}")
        #         print(f"Max Outcome Change: {stat['max_outcome_change']:.4f}")
    
    # End training time measurement
    training_end_time = time()
    total_training_time = training_end_time - training_start_time

    # After training completion, print timing statistics
    print("\nTiming Statistics:")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Average time per epoch: {total_training_time/num_epochs:.2f} seconds")

    # After training, visualize prompt evolution
    visualize_prompt_evolution(model.prompt_manager, output_dir)
    
    # After training completion, print overall statistics
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_gpu_memory_usage = sum(total_gpu_memory) / len(total_gpu_memory)
    
    print("\nTraining Statistics:")
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    print(f"Average GPU memory usage: {avg_gpu_memory_usage:.2f} MB")
    print(f"Peak GPU memory usage: {max(total_gpu_memory):.2f} MB")
    
    # Save metrics to file
    metrics_file = f"{output_dir}/training_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("Training Metrics:\n")
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
        f.write(f"Average epoch time: {avg_epoch_time:.2f} seconds\n")
        f.write(f"Average GPU memory usage: {avg_gpu_memory_usage:.2f} MB\n")
        f.write(f"Peak GPU memory usage: {max(total_gpu_memory):.2f} MB\n")
        f.write("\nEpoch-wise metrics:\n")
        for e, (t, m) in enumerate(zip(epoch_times, total_gpu_memory), 1):
            f.write(f"Epoch {e}: Time={t:.2f}s, GPU Memory={m:.2f}MB\n")

    print(f"\nTraining complete!")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Average epoch time: {total_training_time/num_epochs:.2f} seconds")

    # After training, load best prompts for final evaluation
    print("Loading best model and prompts for final evaluation...")
    model.load_state_dict(torch.load(f'{output_dir}/best_model.pt'))
    model.load_prompts(f'{output_dir}/prompt_state.pt')
    print("\nFinal evaluation on test set:")
    results, _, _, eval_time = evaluate_model(
        model, 
        test_dataloader, 
        device, 
        log,
        output_dir=output_dir,
        use_focal_loss=use_focal_loss,
        use_class_weights=use_class_weights,
        use_prompts=use_prompts,
        use_prompt_updates=use_prompt_updates,
        focal_gamma=focal_gamma,
        mam_flag=mam_flag,
        use_activity_head=use_activity_head,
        use_outcome_head=use_outcome_head,
        task='both'
    )
    total_eval_time = eval_time
    
    # Save comprehensive timing metrics with counter
    timing_file = f"{output_dir}/timing_metrics.txt"
    with open(timing_file, 'w') as f:
        f.write("Timing Metrics:\n")
        f.write("=================\n\n")
        
        # Add GPU Memory Statistics
        gpu_stats = get_detailed_gpu_stats()
        f.write("GPU Memory Usage:\n")
        f.write(f"Allocated VRAM: {gpu_stats['allocated']:.2f} MB\n")
        f.write(f"Cached VRAM: {gpu_stats['cached']:.2f} MB\n")
        f.write(f"Peak VRAM Usage: {gpu_stats['max_allocated']:.2f} MB\n")
        f.write(f"Average VRAM Usage: {avg_gpu_memory_usage:.2f} MB\n\n")
        
        f.write("Training:\n")
        f.write(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Average time per epoch: {total_training_time/num_epochs:.2f} seconds\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Average training time per sample per epoch: {(total_training_time/num_epochs)/len(train_dataset):.4f} seconds\n\n")
        
        f.write("Evaluation:\n")
        f.write(f"Total evaluation time: {total_eval_time:.2f} seconds ({total_eval_time/60:.2f} minutes)\n")
        f.write(f"Evaluation samples: {len(test_dataset)}\n")
        f.write(f"Average evaluation time per sample: {total_eval_time/len(test_dataset):.4f} seconds\n\n")
        
        f.write("Total Processing:\n")
        f.write(f"Total time: {(total_training_time + total_eval_time):.2f} seconds "
                f"({(total_training_time + total_eval_time)/60:.2f} minutes)\n")

    print("\nTiming Statistics Summary:")
    print(f"Training time  : {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Evaluation time: {total_eval_time:.2f} seconds ({total_eval_time/60:.2f} minutes)")
    print(f"Total time     : {(total_training_time + total_eval_time):.2f} seconds "
          f"({(total_training_time + total_eval_time)/60:.2f} minutes)")
    print(f"\nDetailed timing metrics saved to: {timing_file}")

if __name__ == "__main__":
    main()
