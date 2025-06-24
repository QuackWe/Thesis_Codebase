import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ModernBertModel, ModernBertConfig, ModernBertPreTrainedModel, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from collections import Counter
import warnings
import argparse
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Disable TF32 for stability
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def customize_tokenizer_for_activities(df, base_tokenizer):
    """Add activity components as special tokens to the tokenizer"""
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

class MAMDataset(Dataset):
    """Dataset for Masked Activity Modeling that generates prefixes like ProcessTraceDataset"""
    
    def __init__(self, df, tokenizer, activity_encoder, outcome_encoder, customer_type_encoder,
                 max_length=64, mask_prob=0.4, sliding_window=True, window_stride=None, 
                 normalization_params=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.sliding_window = sliding_window
        self.window_stride = window_stride if window_stride is not None else max_length // 2
        self.min_diff = normalization_params[0] if normalization_params else None
        self.max_diff = normalization_params[1] if normalization_params else None
        
        # Use pre-fitted encoders
        self.activity_encoder = activity_encoder
        self.outcome_encoder = outcome_encoder
        self.customer_type_encoder = customer_type_encoder
        
        self.samples = []
        self.masked_tokens = 0
        self.total_tokens = 0
        
        # Process traces and generate prefix samples (same logic as ProcessTraceDataset)
        for trace_id, group in df.groupby('trace_id'):
            group = group.sort_values('TimestampContact')
            activities = group['Activity'].tolist()
            timestamps = group['TimestampContact'].tolist()
            outcome = group['outcome'].iloc[0]
            customer_type = group['type_of_customer'].iloc[0]
            
            # Encode the outcome and customer type
            outcome_encoded = self.outcome_encoder.transform([outcome])[0]
            customer_type_encoded = self.customer_type_encoder.transform([customer_type])[0]
            
            # Calculate time differences
            time_diffs = []
            for i in range(len(timestamps)-1):
                diff = (pd.to_datetime(timestamps[i+1]) - pd.to_datetime(timestamps[i])).total_seconds()
                time_diffs.append(diff)
            
            # Normalize time differences
            if time_diffs:
                time_diffs = np.log1p(time_diffs)
                if self.min_diff is not None and self.max_diff is not None:
                    if self.max_diff > self.min_diff:
                        time_diffs = (time_diffs - self.min_diff) / (self.max_diff - self.min_diff)
                    else:
                        time_diffs = np.zeros_like(time_diffs)
                else:
                    min_diff = np.min(time_diffs)
                    max_diff = np.max(time_diffs)
                    if max_diff > min_diff:
                        time_diffs = (time_diffs - min_diff) / (max_diff - min_diff)
                    else:
                        time_diffs = np.zeros_like(time_diffs)
            
            # Generate all prefixes (minimum length 1) - same as ProcessTraceDataset
            for i in range(1, len(activities)):
                prefix = activities[:i]
                next_activity = activities[i]
                prefix_time_diffs = time_diffs[:i-1]
                
                # Encode the next activity
                next_activity_encoded = self.activity_encoder.transform([next_activity])[0]
                
                # Add padding for time differences
                padded_time_diffs = np.zeros(self.max_length)
                if len(prefix_time_diffs) > 0:
                    n_diffs = min(len(prefix_time_diffs), self.max_length - 1)
                    padded_time_diffs[1:n_diffs+1] = prefix_time_diffs[:n_diffs]
                
                # Format the prefix text with SEP tokens
                prefix_text = ' [SEP] '.join(prefix)
                
                # Tokenize without truncation first
                full_encoding = self.tokenizer(
                    prefix_text,
                    add_special_tokens=True,
                    truncation=False
                )
                
                input_ids_full = full_encoding['input_ids']
                
                # Handle sliding window for long sequences
                if self.sliding_window and len(input_ids_full) > self.max_length:
                    for j in range(0, len(input_ids_full) - self.max_length + 1, self.window_stride):
                        window_ids = input_ids_full[j:j+self.max_length]
                        padded = self.tokenizer.pad(
                            {'input_ids': window_ids},
                            padding='max_length', 
                            max_length=self.max_length
                        )
                        
                        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 
                                          for token_id in padded['input_ids']]
                        
                        self.samples.append({
                            'input_ids': padded['input_ids'],
                            'attention_mask': attention_mask,
                            'time_diffs': padded_time_diffs,
                            'next_activity': next_activity_encoded,
                            'outcome': outcome_encoded,
                            'prefix_length': len(prefix),
                            'customer_type_id': customer_type_encoded
                        })
                else:
                    # Standard truncation and padding
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
                        'next_activity': next_activity_encoded,
                        'outcome': outcome_encoded,
                        'prefix_length': len(prefix),
                        'customer_type_id': customer_type_encoded
                    })
        
        print(f"Created {len(self.samples)} prefix samples for MAM training")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        input_ids = sample['input_ids'].copy()
        attention_mask = sample['attention_mask'].copy()
        
        # Create masked version for MAM
        masked_input_ids = input_ids.copy()
        mam_labels = [-100] * len(input_ids)  # -100 is ignored in loss calculation
        
        # Get vocabulary size for validation
        vocab_size = len(self.tokenizer)
        
        # Define special tokens to avoid masking
        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.mask_token_id
        }
        
        # Find maskable positions (non-special tokens with attention mask = 1)
        maskable_positions = [
            i for i in range(len(input_ids))
            if attention_mask[i] == 1 
            and input_ids[i] not in special_tokens
            and input_ids[i] < vocab_size
        ]
        
        if maskable_positions:
            # Calculate number of tokens to mask (ensure at least 1 for short sequences)
            num_to_mask = max(1, int(len(maskable_positions) * self.mask_prob))
            
            # Randomly select positions to mask
            mask_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
            
            for pos in mask_positions:
                # Store original token ID for loss calculation
                mam_labels[pos] = input_ids[pos]
                
                # Apply 80-10-10 masking strategy
                rand = random.random()
                if rand < 0.8:  # 80% mask token
                    masked_input_ids[pos] = self.tokenizer.mask_token_id
                elif rand < 0.9:  # 10% random token
                    masked_input_ids[pos] = random.randint(0, vocab_size - 1)
                # else 10% unchanged
                
                self.masked_tokens += 1
            
            self.total_tokens += len(maskable_positions)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'masked_input_ids': torch.tensor(masked_input_ids),
            'mam_labels': torch.tensor(mam_labels),
            'time_diffs': torch.tensor(sample['time_diffs'], dtype=torch.float),
            'next_activity': torch.tensor(sample['next_activity']).long(),
            'outcome': torch.tensor(sample['outcome']).long(),
            'prefix_length': torch.tensor(sample['prefix_length']).long(),
            'customer_type_id': torch.tensor(sample['customer_type_id']).long()
        }

class ModernBertMAM(ModernBertPreTrainedModel):
    """ModernBERT model for Masked Activity Modeling"""
    
    def __init__(self, config):
        super().__init__(config)
        self.modernbert = ModernBertModel(config)
        self.mam_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Add process-aware pooler for future fine-tuning compatibility
        self.process_pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        
        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.modernbert.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.modernbert.embeddings.tok_embeddings = value

    def forward(self, input_ids, attention_mask=None):
        outputs = self.modernbert(
            input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.mam_head(sequence_output)
        
        # Also return pooled output for potential future use
        pooled_output = self.process_pooler(sequence_output[:, 0])
        
        return prediction_scores, pooled_output

def compute_mam_class_weights(dataset, tokenizer, device):
    """Compute class weights for MAM based on activity token frequencies"""
    token_counts = Counter()
    
    # Count occurrences of each token in maskable positions
    for sample in dataset.samples:
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        
        # Count only tokens that could be masked (non-special tokens with attention mask = 1)
        special_tokens = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id
        }
        
        for i, token_id in enumerate(input_ids):
            if (attention_mask[i] == 1 and
                token_id not in special_tokens):
                token_counts[token_id] += 1
    
    # Calculate weights with better handling
    total_tokens = sum(token_counts.values())
    num_classes = len(tokenizer)
    
    # Initialize weights for all possible tokens
    weights = torch.ones(num_classes)
    
    # Compute inverse frequency weights with smoothing
    for token_id, count in token_counts.items():
        if token_id < num_classes:
            # Add smoothing factor to prevent extreme weights
            weights[token_id] = total_tokens / (count + 1.0)
    
    # Normalize and clip weights to prevent instability
    weights = weights / weights.sum() * num_classes
    weights = torch.clamp(weights, 0.1, 10.0)
    
    return weights.to(device)

def main():
    # Set device and random seeds for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='Base output directory')
    parser.add_argument('--log', required=True, help='Log name (e.g., mortgages)')
    args = parser.parse_args()
    log = args.log
    
    # Create MAM-specific output directory
    mam_dir = os.path.join(args.output_dir, 'mam_pretrained')
    os.makedirs(mam_dir, exist_ok=True)
    
    # Rest of the code remains the same, just use args.log instead of log
    data_path = f"../datasets/{args.log}/{args.log}_train-val.csv"
    print(f"Loading dataset from {data_path}...")
    
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
        print(f"Successfully loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Initialize tokenizer with activity components
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer = customize_tokenizer_for_activities(df, tokenizer)
    
    # Create and fit label encoders on the entire dataset (same as in model_finetune)
    activity_encoder = LabelEncoder()
    outcome_encoder = LabelEncoder()
    customer_type_encoder = LabelEncoder()
    
    activity_encoder.fit(df['Activity'].unique())
    outcome_encoder.fit(df['outcome'].unique())
    customer_type_encoder.fit(df['type_of_customer'].unique())
    
    print(f"Number of unique activities: {len(activity_encoder.classes_)}")
    print(f"Number of unique outcomes: {len(outcome_encoder.classes_)}")
    print(f"Number of unique customer types: {len(customer_type_encoder.classes_)}")
    
    # Calculate normalization parameters
    normalization_params = calculate_normalization_params(df)
    print(f"Time difference normalization params: min={normalization_params[0]:.4f}, max={normalization_params[1]:.4f}")
    
    # Create MAM dataset with proper encoders
    dataset = MAMDataset(
        df,
        tokenizer,
        activity_encoder,
        outcome_encoder,
        customer_type_encoder,
        max_length=64,
        mask_prob=0.4,  # 40% masking probability as requested
        sliding_window=True,
        window_stride=32,
        normalization_params=normalization_params
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize model
    config = ModernBertConfig.from_pretrained("answerdotai/ModernBERT-base")
    config.vocab_size = len(tokenizer)
    model = ModernBertMAM(config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    # Compute class weights for better loss handling
    class_weights = compute_mam_class_weights(dataset, tokenizer, device)
    
    # Training setup
    num_epochs = 5
    learning_rate = 2e-5
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Add learning rate scheduling
    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    # Loss function with weighted CrossEntropyLoss
    criterion = CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    
    # Mixed precision training for stability
    scaler = torch.cuda.amp.GradScaler()
    
    # Gradient accumulation for stability
    accumulation_steps = 2
    
    print(f"\nStarting MAM pretraining...")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Max length: 64")
    print(f"Masking probability: 40%")
    print(f"Masking strategy: 80-10-10")
    print(f"Masking statistics: {dataset.masked_tokens}/{dataset.total_tokens} tokens masked")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                masked_input_ids = batch['masked_input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['mam_labels'].to(device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs, _ = model(masked_input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    loss = loss / accumulation_steps  # Scale for accumulation
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient accumulation and clipping
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps  # Unscale for logging
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                continue
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Save the pretrained model
    save_path = mam_dir  # Model will be saved directly in mam_pretrained directory
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # Also save the dataset encoders for compatibility with fine-tuning
        encoders = {
            'activity_encoder': activity_encoder,
            'outcome_encoder': outcome_encoder,
            'customer_type_encoder': customer_type_encoder,
            'normalization_params': normalization_params
        }
        torch.save(encoders, os.path.join(save_path, 'encoders.pt'))
        
        print(f"\nModel saved successfully to {save_path}")
        print("Saved files:")
        print("- config.json")
        print("- pytorch_model.bin")
        print("- tokenizer files")
        print("- encoders.pt (for compatibility with fine-tuning)")
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")

if __name__ == "__main__":
    main()
