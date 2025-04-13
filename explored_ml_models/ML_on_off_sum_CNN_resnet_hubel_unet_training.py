# python ML_max_CNN_resnet_hubel_unet_training_2.py train
# python ML_max_CNN_resnet_hubel_unet_training_2.py validate

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

OUT_DIM = 2  # output is now two values: sum of first half (0-4) and sum of second half (5-9) of firing trace
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
# U-Net with Residual Encoder Blocks
#############################################
class ResConvBlock(nn.Module):
    """
    A residual convolution block for the encoder.
    It performs two 3x3 conv layers and adds the shortcut.
    """
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
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

class UNetRegressor(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(UNetRegressor, self).__init__()
        # Encoder with residual blocks
        self.enc1 = ResConvBlock(1, 64)          # Input: (1,200,150) -> (64,200,150)
        self.pool1 = nn.MaxPool2d(2)              # -> (64,100,75)
        self.enc2 = ResConvBlock(64, 128)         # -> (128,100,75)
        self.pool2 = nn.MaxPool2d(2)              # -> (128,50,37)
        self.enc3 = ResConvBlock(128, 256)        # -> (256,50,37)
        self.pool3 = nn.MaxPool2d(2)              # -> (256,25,18) approx.
        
        self.bottleneck = ResConvBlock(256, 512)   # -> (512,25,18)
        
        # Decoder (upsampling with skip connections from encoder)
        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512+256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256+128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128+64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.final_conv = nn.Conv2d(64, OUT_DIM, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)
        p3 = self.pool3(x3)
        bn = self.bottleneck(p3)
        
        # Decoder
        up3 = self.upconv3(bn)
        # Adjust dimensions if necessary
        if up3.size() != x3.size():
            x3 = x3[:, :, :up3.size(2), :up3.size(3)]
        d3_in = torch.cat([up3, x3], dim=1)
        d3 = self.dec3(d3_in)
        
        up2 = self.upconv2(d3)
        if up2.size() != x2.size():
            x2 = x2[:, :, :up2.size(2), :up2.size(3)]
        d2_in = torch.cat([up2, x2], dim=1)
        d2 = self.dec2(d2_in)
        
        up1 = self.upconv1(d2)
        if up1.size() != x1.size():
            x1 = x1[:, :, :up1.size(2), :up1.size(3)]
        d1_in = torch.cat([up1, x1], dim=1)
        d1 = self.dec1(d1_in)
        
        out = self.dropout(d1)
        out = self.final_conv(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
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

def train_pytorch_model(X_train, Y_train, X_test, Y_test, epochs=10, batch_size=32, lr=0.003, dropout_rate=0.3):
    device = get_device()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = UNetRegressor(dropout_rate=dropout_rate).to(device)
    verify_gpu_usage(next(model.parameters()), "Model")
    
    criterion = nn.SmoothL1Loss(beta=0.5)  # Huber loss
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
    # Load X data
    X = np.array(final_X_dict["X"])
    device = get_device()
    n_units = len(selected_keys)
    n_cols = 3
    n_rows = int(np.ceil(n_units / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4), squeeze=False)
    
    for idx, cell_key in enumerate(selected_keys):
        # For the selected cell, load its Y data and compute two values
        Y_all = np.array(final_Y_dict[cell_key])
        Y_first_half = np.max(Y_all[:, 0:5], axis=1, keepdims=True)  # Sum indices 0-4
        Y_second_half = np.max(Y_all[:, 5:10], axis=1, keepdims=True)  # Sum indices 5-9
        Y = np.hstack((Y_first_half, Y_second_half))
        
        # Split data (we use test split only for validation)
        _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
        
        # Load saved model for this cell
        model_path = os.path.join(model_folder, f"unet_model_unit_{cell_key}.pt")
        model = UNetRegressor().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            Y_pred = model(X_test_tensor.to(device))
        Y_test_np = Y_test
        Y_pred_np = Y_pred.cpu().numpy()
        
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        
        # Plot first half predictions in blue
        ax.scatter(Y_test_np[:,0], Y_pred_np[:,0], alpha=0.5, color='blue', label='First half (0-4)')
        # Plot second half predictions in red
        ax.scatter(Y_test_np[:,1], Y_pred_np[:,1], alpha=0.5, color='red', label='Second half (5-9)')
        
        # Plot diagonal line
        if len(Y_test_np) > 0:
            min_val = min(Y_test_np.min(), Y_pred_np.min())
            max_val = max(Y_test_np.max(), Y_pred_np.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--')
            
        ax.set_xlabel('Real Sum')
        ax.set_ylabel('Predicted Sum')
        ax.set_title(f"Cell {cell_key}")
        ax.set_aspect('equal', 'box')
        ax.legend(fontsize='small')
    
    for i in range(n_units, n_rows*n_cols):
        fig.delaxes(axes[i//n_cols][i%n_cols])
    
    fig.suptitle('Validation: Predicted vs. Real (Selected Cells)', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("validation_scatter_subplots.png")
    print("Saved validation plot as validation_scatter_subplots.png")

#############################################
# Main
#############################################
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"  # "train" or "validate"
    
    # Common: load data
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
    
    # Define selected cells (selected_Y_keys as given, zero-indexed)
    # selected_Y_keys = [1, 7, 21, 27, 28, 29, 32, 35, 50, 61, 65, 66, 
    #                    70, 79, 98, 103, 109, 111, 116, 119, 129, 136, 140, 152, 157, 
    #                    161, 165, 166, 167, 169, 170, 171, 178, 179, 180, 181, 184, 
    #                    195, 203, 216, 225, 234]
    # selected_Y_keys = [1, 7, 21, 27, 28, 29, 32, 35, 50, 61, 65, 66, 
    #                    70, 79, 98, 103, 109, 111, 116, 119, 129, 136, 140, 152,]
    #                 #    157, 
    #                 #    161, 165, 166, 167, 169, 170, 171, 178, 179, 180, 181, 184, 
    #                 #    195, 203, 216, 225, 234]
    # selected_Y_keys = [x - 1 for x in selected_Y_keys]
    selected_Y_keys = [0, 26]
    final_Y_dict = {k: final_Y_dict[k] for k in selected_Y_keys}    
    
    if mode == "train":
        # Train models for all cells in final_Y_dict (or a subset)
        for unit_idx, Y_all in final_Y_dict.items():
            Y_all = np.array(Y_all)
            # Create two outputs: sum of first half (0-4) and sum of second half (5-9)
            Y_first_half = np.max(Y_all[:, 0:5], axis=1, keepdims=True)  # Sum indices 0-4
            Y_second_half = np.max(Y_all[:, 5:10], axis=1, keepdims=True)  # Sum indices 5-9
            # Combine them into a single Y array with 2 columns
            Y = np.hstack((Y_first_half, Y_second_half))
            
            print(f"Training model for unit {unit_idx}: X shape {X.shape}, Y shape {Y.shape}")
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            model, history, X_test_tensor, Y_test_tensor = train_pytorch_model(
                X_train, Y_train, X_test, Y_test, epochs=20, batch_size=64, lr=0.0005, dropout_rate=0.3
            )
            plot_training_history(history, unit_idx)
            model_filename = f"unet_model_unit_{unit_idx}.pt"
            torch.save(model.state_dict(), model_filename)
            print(f"Saved model for unit {unit_idx} as {model_filename}")
    elif mode == "validate":
        # Validate only for selected cells.
        model_folder = os.getcwd()  # current directory (or specify folder)
        validate_and_plot(selected_Y_keys, final_X_dict, final_Y_dict, model_folder)
    else:
        print("Unknown mode. Use 'train' or 'validate'.")
        # # do both train and validate
        # for unit_idx, Y_all in final_Y_dict.items():
        #     Y_all = np.array(Y_all)
        #     Y = np.max(Y_all, axis=1, keepdims=True)
        #     print(f"Training model for unit {unit_idx}: X shape {X.shape}, Y shape {Y.shape}")
        #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        #     model, history, X_test_tensor, Y_test_tensor = train_pytorch_model(
        #         X_train, Y_train, X_test, Y_test, epochs=5, batch_size=64, lr=0.003, dropout_rate=0.3
        #     )
        #     plot_training_history(history, unit_idx)
        #     model_filename = f"unet_model_unit_{unit_idx}.pt"
        #     torch.save(model.state_dict(), model_filename)
        #     print(f"Saved model for unit {unit_idx} as {model_filename}")
        #     model_folder = os.getcwd()  # current directory (or specify folder)
        #     validate_and_plot(selected_Y_keys, final_X_dict, final_Y_dict, model_folder)