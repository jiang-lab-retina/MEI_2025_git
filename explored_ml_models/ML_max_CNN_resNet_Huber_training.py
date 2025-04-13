import sys
from os import path
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

OUT_DIM = 1  # output is now the max firing rate (a scalar)
# Force PyTorch to use MPS (Metal) on Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def verify_gpu_usage(tensor, msg=""):
    """Verify that a tensor is actually on the expected device"""
    device_str = str(tensor.device)
    if 'mps' in device_str:
        print(f"✅ {msg} Tensor is on MPS (Apple GPU): {device_str}")
    elif 'cuda' in device_str:
        print(f"✅ {msg} Tensor is on CUDA (NVIDIA GPU): {device_str}")
    else:
        print(f"❌ {msg} Tensor is NOT on GPU: {device_str}")
    return tensor

def load_final_X_Y_dict(final_X_Y_dict_path):
    with open(final_X_Y_dict_path, "rb") as f:
        final_X_Y_dict = pickle.load(f)
    return final_X_Y_dict

# Define a Residual Block for 2D convolutions
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Shortcut for matching dimensions
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Updated CNN model with Residual Blocks and Global Average Pooling
class RGCNet(nn.Module):
    """
    A CNN with residual connections for predicting the max RGC firing rate 
    from visual stimulation, using global average pooling.
    """
    def __init__(self, dropout_rate=0.3):
        super(RGCNet, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 200x150 -> 100x75
        )
        # Residual blocks
        self.resblock1 = ResidualBlock(32, 64, stride=2)  # 100x75 -> ~50x38
        self.resblock2 = ResidualBlock(64, 128, stride=2) # ~50x38 -> ~25x19
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # output: (batch, 128, 1, 1)
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),  # (batch, 128)
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, OUT_DIM)  # output: scalar (max firing rate)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.gap(x)
        x = self.fc(x)
        return x

def get_device():
    """Get the best available device - MPS for Apple Silicon, CUDA for NVIDIA, or CPU"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print(f"✅ Using MPS (Apple Silicon GPU)")
        test_tensor = torch.ones(1, device=device)
        print(f"Test tensor device: {test_tensor.device}")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA (NVIDIA GPU)")
        return device
    else:
        print("❌ GPU not available, using CPU instead")
        return torch.device("cpu")

def train_pytorch_model(X_train, Y_train, X_test, Y_test, epochs=10, batch_size=32, lr=0.003, dropout_rate=0.3):
    """
    Train a PyTorch CNN model for predicting the max RGC firing rate.
    """
    try:
        device = get_device()
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)  # shape: (num_samples, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
        
        print(f"Tensor shapes: X_train: {X_train_tensor.shape}, Y_train: {Y_train_tensor.shape}")
        
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        model = RGCNet(dropout_rate=dropout_rate).to(device)
        verify_gpu_usage(next(model.parameters()), "Model")
        
        # Use Huber loss to reduce sensitivity to extreme errors
        criterion = nn.SmoothL1Loss()  # Huber loss
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_mae += torch.sum(torch.abs(outputs - targets)).item()
            
            train_loss /= len(train_loader.dataset)
            train_mae /= (len(train_loader.dataset) * targets.size(1))
            
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    val_mae += torch.sum(torch.abs(outputs - targets)).item()
                val_loss /= len(test_loader.dataset)
                val_mae /= (len(test_loader.dataset) * targets.size(1))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, history, X_test_tensor, Y_test_tensor
    
    except (RuntimeError, Exception) as e:
        print(f"❌ Error during training: {e}")
        return None, None, None, None

def evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor, unit_idx):
    try:
        device = get_device()
        model.eval()
        if X_test_tensor.device != device:
            X_test_tensor = X_test_tensor.to(device)
        if Y_test_tensor.device != device:
            Y_test_tensor = Y_test_tensor.to(device)
        verify_gpu_usage(X_test_tensor, "Evaluation input")
        with torch.no_grad():
            Y_pred = model(X_test_tensor)
            verify_gpu_usage(Y_pred, "Prediction output")
            mse = nn.MSELoss()(Y_pred, Y_test_tensor).item()
            mae = torch.mean(torch.abs(Y_pred - Y_test_tensor)).item()
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        return Y_pred, mse, mae
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None, 0, 0

def plot_training_history(history, unit_idx):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training and Validation MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"training_history_unit_{unit_idx}.png")
    print(f"Saved training history to training_history_unit_{unit_idx}.png")

def plot_scatter_predictions(Y_true_all, Y_pred_all):
    """
    Create a scatter plot using all test images from each unit (all cells) showing predicted vs real values.
    Y_true_all and Y_pred_all are lists of arrays (each array shape: (N_test, 1)) from each unit.
    """
    true_vals = []
    pred_vals = []
    for true, pred in zip(Y_true_all, Y_pred_all):
        true_vals.extend(true[:, 0])
        pred_vals.extend(pred[:, 0])
    
    plt.figure(figsize=(8, 8))
    plt.scatter(true_vals, pred_vals, alpha=0.7, color='blue', label='Predictions')
    plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--', label='Ideal')
    plt.xlabel('Real Max Firing Rate')
    plt.ylabel('Predicted Max Firing Rate')
    plt.title('Scatter Plot of Predictions vs Real Values (All Test Images)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("scatter_predictions_all_cells.png")
    print("Saved scatter plot as scatter_predictions_all_cells.png")

if __name__ == "__main__":
    print("\n===== GPU AVAILABILITY CHECK =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Apple GPU) available: {torch.backends.mps.is_available()}")
    print(f"MPS (Apple GPU) built: {torch.backends.mps.is_built()}")
    
    try:
        if torch.backends.mps.is_available():
            test_tensor = torch.ones(10, 10, device="mps")
            print(f"✅ Successfully created tensor on MPS: {test_tensor.device}")
            result = test_tensor + test_tensor
            print(f"✅ Basic GPU operation successful: {result.mean().item()}")
            del test_tensor, result
            torch.mps.empty_cache()
        else:
            print("❌ MPS not available, cannot test GPU tensor creation")
    except Exception as e:
        print(f"❌ Error testing GPU: {e}")
    
    print("===== STARTING MAIN PROGRAM =====\n")
    
    try:
        print("Attempting to load real data...")
        all_images_mean = np.load("all_images_mean.npy")
        final_X_dict = load_final_X_Y_dict("final_X_dict.pkl")
        final_Y_dict = load_final_X_Y_dict("final_Y_dict.pkl")
        X = np.array(final_X_dict["X"])
        
        ###########
        # Select Y from final_Y_dict using selected keys
        selected_Y_keys = [1, 7, 21, 27, 28, 29, 32, 35, 50, 61, 65, 66, 
                           70, 79, 98, 103, 109, 111, 116, 119, 129, 136, 140, 152, 157, 
                           161, 165, 166, 167, 169, 170, 171, 178, 179, 180, 181, 184, 
                           195, 203, 216, 225, 234]
        selected_Y_keys = [x - 1 for x in selected_Y_keys]
        Y_all = np.array([final_Y_dict[key] for key in selected_Y_keys])
        ###########
        
        print(f"Loaded real data: X shape {X.shape}, Y_all shape {Y_all.shape}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Could not load real data: {e}")
        exit(1)
    
    # Lists to aggregate predictions and true values from all units
    Y_true_all = []
    Y_pred_all = []
    
    # Loop through each unit; modify Y to be the max firing rate per stimulation.
    for unit_idx in range(Y_all.shape[0]):
        # For each stimulation, take the max over axis 1 to get a scalar (shape: (num_samples, 1))
        Y = np.max(Y_all[unit_idx], axis=1, keepdims=True)
        print(f"Unit {unit_idx}: X shape {X.shape}, Y (max) shape: {Y.shape}")
        print(f"Processing model for unit {unit_idx}")
    
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print(f"Unit {unit_idx}: Train shapes: {X_train.shape}, {Y_train.shape}; Test shapes: {X_test.shape}, {Y_test.shape}")
    
        model, history, X_test_tensor, Y_test_tensor = train_pytorch_model(
            X_train, Y_train, X_test, Y_test, epochs=10, batch_size=64, lr=0.003, dropout_rate=0.3
        )
    
        Y_pred, mse, mae = evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor, unit_idx)
        plot_training_history(history, unit_idx)
    
        model_filename = f"rgc_model_unit_{unit_idx}.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")
    
        Y_true_all.append(Y_test_tensor.cpu().numpy())
        Y_pred_all.append(Y_pred.cpu().numpy())
    
    plot_scatter_predictions(Y_true_all, Y_pred_all)