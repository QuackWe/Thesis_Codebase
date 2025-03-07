from MTLFormer import MTLFormer, train_model
from evaluate import evaluate_model
import torch
from torch import optim
from dataloader import train_loader, val_loader

# Model hyperparameters
embed_size = 235
heads = 5
dropout = 0.3
num_classes = 235  # For next activity prediction

# Instantiate model
model = MTLFormer(embed_size, heads, dropout, num_classes)

# Check if CUDA is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print('Device: ', device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Define loss weights (tuning these may help balance task performance)
weights = [0.6, 0.2, 0.2]  # Can be tuned

# Train the model
epoch, num_epochs, total_loss = train_model(model, train_loader, optimizer, weights, num_epochs=100)
print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

# Evaluate the model
accuracy, avg_time_mae, avg_remaining_mae = evaluate_model(model, val_loader)
print(f'Accuracy for next activity: {accuracy:.4f}')
print(f'MAE for next event time: {avg_time_mae:.4f}')
print(f'MAE for remaining time: {avg_remaining_mae:.4f}')