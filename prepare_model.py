import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os

from preprocessing import preprocess_timeseries_dataframe
from model import MyModel
from training import train
from dataset import RepoSplitTimeSeriesDataset
from torch.utils.data import DataLoader # Import DataLoader

# Define paths
DATA_PATH = 'resources/data_v3.csv'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'encoders.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Apply initial preprocessing steps from the notebook
from dateutil import parser

def parse_mixed_date(date_str):
    try:
        return parser.parse(date_str, dayfirst=True)
    except Exception:
        return pd.NaT

df['Scan date'] = df['Scan date'].astype(str).apply(parse_mixed_date)
df['Scan date'] = pd.to_datetime(df['Scan date'], errors='coerce')

df['Commit frequency'] = df['Commit frequency'].str.capitalize()

label_map = {
    'Very low': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very high': 4
}
df['Evaluate'] = df["Evaluate"].map(label_map)

# Preprocess data
print("Preprocessing data...")
X, y, meta, encoders, scaler = preprocess_timeseries_dataframe(df)

# Save encoders and scaler
print(f"Saving encoders to {ENCODERS_PATH} and scaler to {SCALER_PATH}...")
with open(ENCODERS_PATH, 'wb') as f:
    pickle.dump(encoders, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

# Prepare datasets and dataloaders
timesteps = 5
split_ratio = 0.8

print(f"Creating train and validation datasets with timesteps={timesteps} and split_ratio={split_ratio}...")
train_dataset = RepoSplitTimeSeriesDataset(X, y, meta, timesteps=timesteps, mode="train", split_ratio=split_ratio)
val_dataset = RepoSplitTimeSeriesDataset(X, y, meta, timesteps=timesteps, mode="val", split_ratio=split_ratio)

batch_size = 64
print(f"Creating DataLoaders with batch_size={batch_size}...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and criterion
input_features = X.shape[1]
num_classes = len(y.unique()) 

print(f"Initializing model with input_features={input_features}, timesteps={timesteps}, num_classes={num_classes}...")
model = MyModel(input_features=input_features, timesteps=timesteps, num_classes=num_classes)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Train the model
epochs = 100
print(f"Starting model training for {epochs} epochs...")
train_losses, val_accuracies = train(
    model, train_loader, optimizer, criterion,
    val_loader=val_loader, epochs=epochs
)

print("Training complete.")

# Save the trained model
print(f"Saving trained model to {MODEL_PATH}...")
torch.save(model.state_dict(), MODEL_PATH)

print("Model training and saving process finished.")
