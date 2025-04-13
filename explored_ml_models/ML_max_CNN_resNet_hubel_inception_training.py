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

# Define an Inception module
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
        super(InceptionModule, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(out1x1),
            nn.ReLU(inplace=True)
        )
        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1, bias=False),
            nn.BatchNorm2d(red3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out3x3),
            nn.ReLU(inplace=True)
        )
        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size=1, bias=False),
            nn.BatchNorm2d(red5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(red5x5, out5x5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out5x5),
            nn.ReLU(inplace=True)
        )
        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# Updated CNN model using Inception modules and Global Average Pooling
class RGCNet(nn.Module):
    """
    A CNN using Inception modules for predicting the max RGC firing rate 
    from visual stimulation.
    """
    def __init__(self, dropout_rate=0.3):
        super(RGCNet, self).__init__()
        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 200x150 -> 100x75
        )
        # Inception Module 1: Input channels = 32, output channels = 32+64+16+16 = 128.
        self.inception1 = InceptionModule(in_channels=32,
                                          out1x1=32,
                                          red3x3=48, out3x3=64,
                                          red5x5=8,  out5x5=16,
                                          pool_proj=16)
        # Inception Module 2: Input channels = 128, output channels = 64+96+32+32 = 224.
        self.inception2 = InceptionModule(in_channels=128,
                                          out1x1=64,
                                          red3x3=64, out3x3=96,
                                          red5x5=16, out5x5=32,
                                          pool_proj=32)
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # output: (batch, 224, 1, 1)
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),  # (batch, 224)
            nn.Linear(224, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, OUT_DIM)  # output: scalar (max firing rate)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
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
        
        # Use Huber loss (SmoothL1Loss)
        criterion = nn.SmoothL1Loss()
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

def plot_scatter_predictions(Y_true_all, Y_pred_all, selected_keys):
    """
    Create a figure with subplots for each cell.
    Each subplot shows a scatter plot of predicted vs. real values for all test images from that cell.
    The subtitle of each subplot is set to the corresponding key from selected_keys.
    """
    n_units = len(Y_true_all)
    n_cols = 5
    n_rows = int(np.ceil(n_units / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
    
    for i in range(n_units):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col]
        true_vals = Y_true_all[i][:, 0]
        pred_vals = Y_pred_all[i][:, 0]
        ax.scatter(true_vals, pred_vals, alpha=0.5, color='blue')
        ax.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')
        ax.set_xlabel('Real')
        ax.set_ylabel('Predicted')
        ax.set_title(f"Cell {selected_keys[i]}")
    
    # Hide any empty subplots if n_units is not a multiple of n_cols
    for i in range(n_units, n_rows * n_cols):
        fig.delaxes(axes[i // n_cols][i % n_cols])
    
    fig.suptitle('Predicted vs. Real Max Firing Rate (Validation)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("scatter_subplots_all_cells.png")
    print("Saved scatter subplot figure as scatter_subplots_all_cells.png")

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
        
        # Select Y from final_Y_dict using selected keys
        selected_Y_keys = [1, 7, 21, 27, 28, 29, 32, 35, 50, 61, 65, 66, 
                           70, 79, 98, 103, 109, 111, 116, 119, 129, 136, 140, 152, 157, 
                           161, 165, 166, 167, 169, 170, 171, 178, 179, 180, 181, 184, 
                           195, 203, 216, 225, 234]
        # Convert to zero-indexed keys
        selected_Y_keys = [x - 1 for x in selected_Y_keys]
        Y_all = np.array([final_Y_dict[key] for key in selected_Y_keys])
        
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
    
    # Create subplots for each cell using the selected_Y_keys as subtitles
    plot_scatter_predictions(Y_true_all, Y_pred_all, selected_Y_keys)