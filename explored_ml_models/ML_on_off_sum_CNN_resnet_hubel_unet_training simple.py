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

# Function to apply average pooling to reduce image dimensions
def reduce_image_size(images, target_height=25, target_width=15):
    """
    Reduce image dimensions using average pooling
    
    Args:
        images: NumPy array of shape (N, H, W)
        target_height: Target height after reduction
        target_width: Target width after reduction
    
    Returns:
        Reduced images as NumPy array of shape (N, target_height, target_width)
    """
    # Convert to tensor for pooling operation
    images_tensor = torch.tensor(images, dtype=torch.float32)
    
    # Calculate pool kernel size and stride
    orig_height, orig_width = images.shape[1:3]
    kernel_height = int(orig_height / target_height)
    kernel_width = int(orig_width / target_width)
    
    # Create average pooling layer
    avg_pool = nn.AvgPool2d(kernel_size=(kernel_height, kernel_width), 
                           stride=(kernel_height, kernel_width))
    
    # Apply pooling (add batch dimension then remove it)
    images_tensor = images_tensor.unsqueeze(1)  # Add channel dimension
    pooled_images = avg_pool(images_tensor)
    pooled_images = pooled_images.squeeze(1)  # Remove channel dimension
    
    # Convert back to numpy
    return pooled_images.numpy()

#############################################
# U-Net with Residual Encoder Blocks
#############################################
class ResConvBlock(nn.Module):
    """
    A simplified residual convolution block with fewer parameters.
    """
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        # Single conv layer with smaller kernel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Simpler shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.bn(self.conv(x))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class UNetRegressor(nn.Module):
    def __init__(self, dropout_rate=0.5):  # Increased dropout rate
        super(UNetRegressor, self).__init__()
        # Encoder with fewer channels and simplified blocks
        self.enc1 = ResConvBlock(1, 32)          # Reduced from 64 to 32 channels
        self.pool1 = nn.MaxPool2d(2)             # -> (32,12,7)
        self.enc2 = ResConvBlock(32, 64)         # Reduced from 128 to 64 channels
        self.pool2 = nn.MaxPool2d(2)             # -> (64,6,3)
        
        # Bottleneck with fewer channels
        self.bottleneck = ResConvBlock(64, 128)  # Reduced from 256 to 128 channels
        
        # Decoder - simplified for smaller input size
        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128+64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added BatchNorm for stability
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate/2)  # Added spatial dropout
        )
        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64+32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added BatchNorm for stability
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate/2)  # Added spatial dropout
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.final_conv = nn.Conv2d(32, OUT_DIM, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        bn = self.bottleneck(p2)
        
        # Decoder - simplified
        up2 = self.upconv2(bn)
        # Adjust dimensions if necessary
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

def train_pytorch_model(X_train, Y_train, X_test, Y_test, epochs=10, batch_size=32, lr=0.003, dropout_rate=0.5):
    device = get_device()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    
    # Print model size information
    model = UNetRegressor(dropout_rate=dropout_rate).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} total parameters, {trainable_params:,} trainable")
    verify_gpu_usage(next(model.parameters()), "Model")
    
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    criterion = nn.SmoothL1Loss(beta=0.5)  # Huber loss
    # Added weight decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=10, verbose=True
    )
    
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    best_model_state = None
    
    # Early stopping
    patience = 15
    epochs_no_improve = 0
    
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
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        # Step the learning rate scheduler
        scheduler.step(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # More informative print
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, lr: {current_lr:.6f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch with val_loss: {best_val_loss:.4f}")
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
    # Reduce image size
    X = reduce_image_size(X)
    
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
        ax.scatter(Y_test_np[:,0], Y_pred_np[:,0], alpha=0.05, color='blue', label='First half (0-4)')
        # Plot second half predictions in red
        ax.scatter(Y_test_np[:,1], Y_pred_np[:,1], alpha=0.05, color='red', label='Second half (5-9)')
        # Create boxplots for each unique value of Y_test_np
        # First half (0-4) boxplots
        unique_y_first = np.unique(Y_test_np[:, 0])
        if len(unique_y_first) > 0:
            boxplot_data_first = []
            boxplot_positions_first = []
            
            for val in unique_y_first:
                # Find all predictions where the actual value equals this unique value
                mask = Y_test_np[:, 0] == val
                if np.sum(mask) > 0:  # Only include if we have data points
                    boxplot_data_first.append(Y_pred_np[mask, 0])
                    boxplot_positions_first.append(val)
            
            # Add boxplot for first half data (blue)
            if boxplot_data_first:
                bp1 = ax.boxplot(boxplot_data_first, positions=boxplot_positions_first, 
                                widths=0.2, patch_artist=True, vert=True)
                for patch in bp1['boxes']:
                    patch.set_facecolor('blue')
                    patch.set_alpha(1)
                for whisker in bp1['whiskers']:
                    whisker.set(color='blue', alpha=1)
                for cap in bp1['caps']:
                    cap.set(color='blue', alpha=1)
                for median in bp1['medians']:
                    median.set(color='blue', linewidth=2)
        
        # Second half (5-9) boxplots
        unique_y_second = np.unique(Y_test_np[:, 1])
        if len(unique_y_second) > 0:
            boxplot_data_second = []
            boxplot_positions_second = []
            
            for val in unique_y_second:
                # Find all predictions where the actual value equals this unique value
                mask = Y_test_np[:, 1] == val
                if np.sum(mask) > 0:  # Only include if we have data points
                    boxplot_data_second.append(Y_pred_np[mask, 1])
                    boxplot_positions_second.append(val)
            
            # Add boxplot for second half data (red)
            if boxplot_data_second:
                bp2 = ax.boxplot(boxplot_data_second, positions=boxplot_positions_second, 
                                widths=0.2, patch_artist=True, vert=True)
                for patch in bp2['boxes']:
                    patch.set_facecolor('red')
                    patch.set_alpha(1)
                for whisker in bp2['whiskers']:
                    whisker.set(color='red', alpha=1)
                for cap in bp2['caps']:
                    cap.set(color='red', alpha=1)
                for median in bp2['medians']:
                    median.set(color='red', linewidth=2)

        
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
        
        # Reduce image size
        X = reduce_image_size(X)
        print(f"Reduced X shape: {X.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Define selected cells (selected_Y_keys as given, zero-indexed)
    selected_Y_keys = [1, 7, 21, 27, 28, 29, 32, 35, 50, 61, 65, 66, 
                       70, 79, 98, 103, 109, 111, 116, 119, 129, 136, 140, 152, 157, 
                       161, 165, 166, 167, 169, 170, 171, 178, 179, 180, 181, 184, 
                       195, 203, 216, 225, 234]
    # selected_Y_keys = [1, 7, 21, 27, 28, 29, 32, 35, 50, 61, 65, 66, 
    #                    70, 79, 98, 103, 109, 111, 116, 119, 129, 136, 140, 152,]
    #                 #    157, 
    #                 #    161, 165, 166, 167, 169, 170, 171, 178, 179, 180, 181, 184, 
    #                 #    195, 203, 216, 225, 234]
    selected_Y_keys = [x - 1 for x in selected_Y_keys]
    #selected_Y_keys = [0, 26]
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
                X_train, Y_train, X_test, Y_test, epochs=200, batch_size=64, lr=0.001, dropout_rate=0.3
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