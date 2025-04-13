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
import torchvision.models as models
import torch.nn.functional as F
import segmentation_models_pytorch as smp  # ensure this package is installed

OUT_DIM = 1  # output is the max firing rate (a scalar)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def verify_gpu_usage(tensor, msg=""):
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
        return pickle.load(f)

#############################################
# Pretrained U-Net Regressor using ResNet50 encoder
#############################################
class ResNet50UNetRegressor(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ResNet50UNetRegressor, self).__init__()
        # Create a pretrained U-Net with ResNet50 encoder using SMP
        self.unet = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,   # our input is replicated to 3 channels
            classes=1        # output channel for segmentation/regression head
        )
        # Global average pooling and regression head
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1, OUT_DIM)
    
    def forward(self, x):
        # x: (batch,1,H,W) -> replicate to 3 channels and resize to 224x224
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        seg_out = self.unet(x)  # shape: (batch, 1, H, W)
        gap = self.gap(seg_out) # shape: (batch, 1, 1, 1)
        gap = gap.view(gap.size(0), -1)  # (batch, 1)
        gap = self.dropout(gap)
        out = self.fc(gap)  # (batch, OUT_DIM)
        return out

#############################################
# Training and Evaluation Functions
#############################################
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print(f"✅ Using MPS (Apple GPU)")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA (NVIDIA GPU)")
        return device
    else:
        print("❌ GPU not available, using CPU instead")
        return torch.device("cpu")

def train_pytorch_model(X_train, Y_train, X_test, Y_test, epochs=10, batch_size=32, lr=0.001, dropout_rate=0.3):
    device = get_device()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)  # shape: (N,1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = ResNet50UNetRegressor(dropout_rate=dropout_rate).to(device)
    verify_gpu_usage(next(model.parameters()), "Model")
    
    criterion = nn.SmoothL1Loss()  # Huber loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for inputs, targets in train_loader:
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
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, history, X_test_tensor, Y_test_tensor

def evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor):
    device = get_device()
    model.eval()
    X_test_tensor = X_test_tensor.to(device)
    Y_test_tensor = Y_test_tensor.to(device)
    with torch.no_grad():
        Y_pred = model(X_test_tensor)
        mse = nn.MSELoss()(Y_pred, Y_test_tensor).item()
        mae = torch.mean(torch.abs(Y_pred - Y_test_tensor)).item()
    print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
    return Y_pred, mse, mae

def plot_training_history(history, unit_idx):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"training_history_unit_{unit_idx}.png")
    print(f"Saved training history for unit {unit_idx}")

#############################################
# Independent Validation Plot
#############################################
def validate_and_plot(selected_keys, final_X_dict, final_Y_dict, model_folder):
    X = np.array(final_X_dict["X"])
    device = get_device()
    n_units = len(selected_keys)
    n_cols = 3
    n_rows = int(np.ceil(n_units / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4), squeeze=False)
    
    for idx, cell_key in enumerate(selected_keys):
        Y_all = np.array(final_Y_dict[cell_key])
        Y = np.max(Y_all, axis=1, keepdims=True)
        # Use test split for validation
        _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
        
        model_path = os.path.join(model_folder, f"resnet50_unet_model_unit_{cell_key}.pt")
        model = ResNet50UNetRegressor(dropout_rate=0.3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            Y_pred = model(X_test_tensor.to(device))
        Y_test_np = Y_test
        Y_pred_np = Y_pred.cpu().numpy()
        
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        ax.scatter(Y_test_np[:,0], Y_pred_np[:,0], alpha=0.5, color='blue')
        if len(Y_test_np) > 0:
            ax.plot([Y_test_np[:,0].min(), Y_test_np[:,0].max()],
                    [Y_test_np[:,0].min(), Y_test_np[:,0].max()], 'r--')
        ax.set_xlabel('Real')
        ax.set_ylabel('Predicted')
        ax.set_title(f"Cell {cell_key}")
    
    for i in range(n_units, n_rows*n_cols):
        fig.delaxes(axes[i//n_cols][i%n_cols])
    
    fig.suptitle('Validation: Predicted vs Real (Selected Cells)', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("validation_scatter_subplots.png")
    print("Saved validation plot as validation_scatter_subplots.png")

#############################################
# Main
#############################################
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"  # "train" or "validate"
    
    try:
        print("Loading real data...")
        all_images_mean = np.load("all_images_mean.npy")
        final_X_dict = load_final_X_Y_dict("final_X_dict.pkl")
        final_Y_dict = load_final_X_Y_dict("final_Y_dict.pkl")
        X = np.array(final_X_dict["X"])
        print(f"Loaded X shape: {X.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Define selected cells (zero-indexed)
    # selected_Y_keys = [1, 7, 21, 27, 28, 29, 32, 35, 50, 61, 65, 66, 70, 79, 98, 103, 109, 111, 116, 119, 129, 136, 140, 152]
    # selected_Y_keys = [x - 1 for x in selected_Y_keys]
    selected_Y_keys = [0]
    final_Y_dict = {k: final_Y_dict[k] for k in selected_Y_keys}    
    
    if mode == "train":
        for unit_idx, Y_all in final_Y_dict.items():
            Y_all = np.array(Y_all)
            Y = np.max(Y_all, axis=1, keepdims=True)
            print(f"Training model for unit {unit_idx}: X shape {X.shape}, Y shape {Y.shape}")
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            model, history, X_test_tensor, Y_test_tensor = train_pytorch_model(
                X_train, Y_train, X_test, Y_test, epochs=10, batch_size=64, lr=0.001, dropout_rate=0.3
            )
            plot_training_history(history, unit_idx)
            model_filename = f"resnet50_unet_model_unit_{unit_idx}.pt"
            torch.save(model.state_dict(), model_filename)
            print(f"Saved model for unit {unit_idx} as {model_filename}")
    elif mode == "validate":
        model_folder = os.getcwd()
        final_Y_dict = {k: final_Y_dict[k] for k in selected_Y_keys}
        validate_and_plot(selected_Y_keys, final_X_dict, final_Y_dict, model_folder)
    else:
        print("Unknown mode. Use 'train' or 'validate'.")