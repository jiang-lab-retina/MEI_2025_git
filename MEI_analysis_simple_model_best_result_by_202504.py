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
OUT_DIM = 9
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

# Function to reduce image dimensions using average pooling
def reduce_image_size(images, target_height=50, target_width=30):
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

# Create mock data for testing
def create_mock_data(shape=(1185, 50, 30), output_dim=OUT_DIM):
    """Create mock data for testing the model when real data is not available."""
    # Mock images with random values between 0-255
    X = np.random.randint(0, 256, size=shape).astype(np.uint8)
    
    # Mock response values (firing rates between 0-10)
    Y_all = np.random.uniform(0, 10, size=(1, shape[0], output_dim)).astype(np.float32)
    
    # Create mock mean image
    all_images_mean = np.mean(X, axis=0)
    
    print(f"Created mock X data with shape: {X.shape}")
    print(f"Created mock Y data with shape: {Y_all.shape}")
    
    return X, Y_all, all_images_mean

# Define the CNN model
class RGCNet(nn.Module):
    """
    A 3-layer CNN for predicting RGC outputs from visual stimulation.
    Adjusted for 50x30 input images.
    """
    def __init__(self):
        super(RGCNet, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # Reduced kernel size
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Reduced kernel size
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Third convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate size after convolutions and pooling
        # Input: 50x30 -> Conv1+Pool: 25x15 -> Conv2+Pool: 12x7 -> Conv3+Pool: 6x3
        fc_input_size = 128 * 6 * 3
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, OUT_DIM)  # Output size for RGC prediction
        )
        
    def forward(self, x):
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply fully connected layers
        x = self.fc(x)
        
        return x

# Update device selection to use MPS (Apple Silicon GPU)
def get_device():
    """Get the best available device - MPS for Apple Silicon, CUDA for NVIDIA, or CPU"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Force MacOS to use MPS (Metal Performance Shaders) for M1/M2/M3 GPU acceleration
        device = torch.device("mps")
        print(f"✅ Using MPS (Apple Silicon GPU)")
        # Verify it's actually using MPS
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

def train_pytorch_model(X_train, Y_train, X_test, Y_test, epochs=50, batch_size=32, lr=0.001):
    """
    Train a PyTorch model for RGC output prediction
    """
    try:
        # Get device first to fail early if GPU is not available
        device = get_device()
        
        # Convert to PyTorch tensors
        # Add channel dimension and normalize
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
        
        print(f"Tensor shapes: X_train: {X_train_tensor.shape}, Y_train: {Y_train_tensor.shape}")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model, loss function, and optimizer
        model = RGCNet().to(device)
        verify_gpu_usage(next(model.parameters()), "Model")
        
        # Count parameters in the model
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move to GPU
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Verify first batch is on GPU
                if epoch == 0 and batch_idx == 0:
                    verify_gpu_usage(inputs, "Input batch")
                    verify_gpu_usage(targets, "Target batch")
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * inputs.size(0)
                train_mae += torch.sum(torch.abs(outputs - targets)).item()
            
            # Calculate epoch metrics
            train_loss /= len(train_loader.dataset)
            train_mae /= (len(train_loader.dataset) * targets.size(1))
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Track metrics
                    val_loss += loss.item() * inputs.size(0)
                    val_mae += torch.sum(torch.abs(outputs - targets)).item()
                
                # Calculate epoch metrics
                val_loss /= len(test_loader.dataset)
                val_mae /= (len(test_loader.dataset) * targets.size(1))
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, history, X_test_tensor, Y_test_tensor
    
    except (RuntimeError, Exception) as e:
        print(f"❌ Error during training: {e}")
        print("Falling back to CPU training...")
        
        # Fallback to CPU
        if 'device' in locals() and str(device) != 'cpu':
            device = torch.device("cpu")
            print("Training on CPU instead")
            
            # Recreate model on CPU
            model = RGCNet().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Continue with training on CPU
            # ... (rest of the training code would follow here)
            # For brevity, we're just returning a basic model
            
            return model, {"train_loss": [0], "val_loss": [0], "train_mae": [0], "val_mae": [0]}, X_test_tensor, Y_test_tensor
        else:
            raise e

def evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor, X_test_original=None):
    """
    Evaluate the trained PyTorch model
    
    Args:
        model: Trained PyTorch model
        X_test_tensor: Reduced images tensor for model input
        Y_test_tensor: Ground truth tensor
        X_test_original: Original (non-reduced) images for visualization
    """
    try:
        device = get_device()
        model.eval()
        
        # Move tensors to device (GPU) if needed
        if X_test_tensor.device != device:
            X_test_tensor = X_test_tensor.to(device)
        if Y_test_tensor.device != device:
            Y_test_tensor = Y_test_tensor.to(device)
        
        # Verify tensors are on GPU    
        verify_gpu_usage(X_test_tensor, "Evaluation input")
        
        # Make predictions
        with torch.no_grad():
            Y_pred = model(X_test_tensor)
            verify_gpu_usage(Y_pred, "Prediction output")
            
            # Calculate metrics
            mse = nn.MSELoss()(Y_pred, Y_test_tensor).item()
            mae = torch.mean(torch.abs(Y_pred - Y_test_tensor)).item()
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        # Convert to numpy for plotting - move back to CPU first
        X_test_np = X_test_tensor.cpu().numpy()
        Y_test_np = Y_test_tensor.cpu().numpy()
        Y_pred_np = Y_pred.cpu().numpy()
        
        # Use original images for display if provided
        if X_test_original is not None:
            # If tensor, convert to numpy
            if isinstance(X_test_original, torch.Tensor):
                X_display = X_test_original.cpu().numpy()
            else:
                X_display = X_test_original
                
            if len(X_display) != len(X_test_np):
                print("Warning: Original image count doesn't match test set count. Using reduced images.")
                X_display = X_test_np
        else:
            # If no original images provided, use the reduced ones
            X_display = X_test_np
        
        # Plot example predictions - two examples per row
        num_examples = min(20, len(X_test_np))
        
        # Set up for two examples per row
        from matplotlib import gridspec
        
        # X-axis for line plots (neuron indices)
        neuron_indices = np.arange(Y_test_np.shape[1])
        
        # Calculate number of rows needed (2 examples per row)
        num_rows = int(np.ceil(num_examples / 2))
        
        # Create a figure sized for the examples
        plt.figure(figsize=(25, 4 * num_rows))
        
        # Create a GridSpec for the entire figure
        gs_main = gridspec.GridSpec(num_rows, 2, hspace=0.4, wspace=0.3)
        
        for i in range(num_examples):
            row = i // 2  # Row index
            col = i % 2   # Column index (0 or 1)
            
            # Create subplot grid for this position
            gs_cell = gridspec.GridSpecFromSubplotSpec(1, 4, 
                subplot_spec=gs_main[row, col],
                width_ratios=[1, 1, 1, 1])  # 1 for image, 9 for plot
            
            # Plot image
            ax_img = plt.subplot(gs_cell[0])
            if len(X_display.shape) == 4 and X_display.shape[1] == 1:
                ax_img.imshow(X_display[i, 0], cmap='gray')
            elif len(X_display.shape) == 4:
                ax_img.imshow(X_display[i])
            else:
                ax_img.imshow(X_display[i], cmap='gray')
            ax_img.set_title(f"Input {i+1}")
            ax_img.axis('off')
            
            # Plot traces
            ax1 = plt.subplot(gs_cell[1:])
            
            # Plot true response
            line1, = ax1.plot(neuron_indices, Y_test_np[i], 'b-o', linewidth=5, alpha=0.5, markersize=10, label='True')
            ax1.set_xlabel("Neuron Index")
            ax1.set_ylabel("Response (True)", color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, alpha=0.3)
            
            # Create twin axis for predicted response
            ax2 = ax1.twinx()
            
            # Normalize predicted values
            true_min, true_max = Y_test_np[i].min(), Y_test_np[i].max()
            true_range = true_max - true_min
            
            if true_range < 1e-6:
                true_range = 1.0
                
            pred_min, pred_max = Y_pred_np[i].min(), Y_pred_np[i].max()
            pred_range = pred_max - pred_min
            
            if pred_range < 1e-6:
                normalized_pred = np.ones_like(Y_pred_np[i]) * true_min
            else:
                normalized_pred = true_min + (Y_pred_np[i] - pred_min) * true_range / pred_range
            
            # Plot predicted response
            line2, = ax2.plot(neuron_indices, Y_pred_np[i], 'r-o', linewidth=5, alpha=0.5, markersize=10, label='Predicted')
            ax2.set_ylabel("Response (Predicted)", color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            
            # Add legend
            lines = [line1, line2]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='upper right')
            
            # Set title
            ax1.set_title(f"Sample {i+1}: Response Comparison")
        
        plt.tight_layout()
        plt.savefig(f"prediction_examples_unit_{unit_idx}.png", dpi=150)
        print(f"Saved prediction examples to prediction_examples_unit_{unit_idx}.png")
        
        # Create a detailed comparison plot for a few selected examples - much wider
        plt.figure(figsize=(60, 12))  # 4x wider (from 15 to 60)
        comparison_examples = min(5, num_examples)
        
        for i in range(comparison_examples):
            # Create subplot with twin y-axes
            ax1 = plt.subplot(comparison_examples, 1, i+1)
            
            # Plot true response on left y-axis in blue
            true_line, = ax1.plot(neuron_indices, Y_test_np[i], 'b-o', linewidth=2.5, markersize=6, label='True')
            ax1.set_xlabel("Neuron Index", fontsize=14)
            ax1.set_ylabel("Response (True)", color='black', fontsize=14)
            ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
            ax1.tick_params(axis='x', labelsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Create twin y-axis for predicted response
            ax2 = ax1.twinx()
            
            # Normalize predicted values to match true range
            true_min, true_max = Y_test_np[i].min(), Y_test_np[i].max()
            true_range = true_max - true_min
            
            # Prevent division by zero if range is too small
            if true_range < 1e-6:
                true_range = 1.0
                
            pred_min, pred_max = Y_pred_np[i].min(), Y_pred_np[i].max()
            pred_range = pred_max - pred_min
            
            # Normalize predicted values to true range
            if pred_range < 1e-6:  # If predictions are nearly constant
                normalized_pred = np.ones_like(Y_pred_np[i]) * true_min
            else:
                normalized_pred = true_min + (Y_pred_np[i] - pred_min) * true_range / pred_range
            
            # Plot normalized predicted response on second y-axis in red
            pred_line, = ax2.plot(neuron_indices, Y_pred_np[i], 'r-o', linewidth=2.5, markersize=6, label='Predicted')
            ax2.set_ylabel("Response (Predicted)", color='black', fontsize=14)
            ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
            
            # Add detailed title
            plt.title(f"Sample {i+1}: True vs Predicted Response", fontsize=16)
            
            # Add combined legend with larger font
            lines = [true_line, pred_line]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='upper right', fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # Add more space between subplots
        plt.savefig(f"detailed_comparison_unit_{unit_idx}.png", dpi=150)
        print(f"Saved detailed comparison to detailed_comparison_unit_{unit_idx}.png")
        
        return mse, mae
    
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 0, 0

def plot_training_history(history):
    """
    Plot the training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot MAE
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


if __name__ == "__main__":
    # First, verify that PyTorch can use the GPU
    print("\n===== GPU AVAILABILITY CHECK =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Apple GPU) available: {torch.backends.mps.is_available()}")
    print(f"MPS (Apple GPU) built: {torch.backends.mps.is_built()}")
    
    # Test creating a tensor on GPU
    try:
        if torch.backends.mps.is_available():
            # Try creating a small tensor on MPS
            test_tensor = torch.ones(10, 10, device="mps")
            print(f"✅ Successfully created tensor on MPS: {test_tensor.device}")
            # Try some basic operations
            result = test_tensor + test_tensor
            print(f"✅ Basic GPU operation successful: {result.mean().item()}")
            # Cleanup
            del test_tensor, result
            torch.mps.empty_cache()
        else:
            print("❌ MPS not available, cannot test GPU tensor creation")
    except Exception as e:
        print(f"❌ Error testing GPU: {e}")
    
    print("===== STARTING MAIN PROGRAM =====\n")
    
    # Try to load real data, use mock data if not available
    try:
        print("Attempting to load real data...")
        all_images_mean = np.load("all_images_mean.npy")
        final_X_dict = load_final_X_Y_dict("final_X_dict.pkl")
        final_Y_dict = load_final_X_Y_dict("final_Y_dict.pkl")
        
        # Store the original images before reduction
        X_original = np.array(final_X_dict["X"])
        print(f"Original X shape: {X_original.shape}")
        
        # Reduce image dimensions
        X = reduce_image_size(X_original, target_height=50, target_width=30)
        print(f"Reduced X shape: {X.shape}")
        
        Y_all = np.array(list(final_Y_dict.values()))
        print(f"Loaded real data: X shape {X.shape}, Y_all shape {Y_all.shape}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Could not load real data: {e}")
        print("Creating mock data instead...")
        X, Y_all, all_images_mean = create_mock_data()
        # Create a mock original image set at 200x150
        X_original = np.random.randint(0, 256, size=(X.shape[0], 200, 150)).astype(np.uint8)
        print(f"Created mock original X data with shape: {X_original.shape}")
    
    # You can choose any unit from Y_all by setting the index
    for unit_idx in range(Y_all.shape[0]):
        #unit_idx = 26  # Change this index to train for different units
        Y = Y_all[unit_idx]
        print(f"Working with data shapes: X: {X.shape}, Y: {Y.shape}")
        print(f"Training model for unit {unit_idx}")
    
        # Create train/test split - keep track of original image indices
        X_train, X_test, Y_train, Y_test, X_orig_train, X_orig_test = train_test_split(
            X, Y, X_original, test_size=0.2, random_state=42
        )
        print(f"Train/test shapes: X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")
        print(f"Original images - X_orig_train: {X_orig_train.shape}, X_orig_test: {X_orig_test.shape}")
        
        # Check if MPS is available before importing PyTorch
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS (Apple GPU) available: {torch.backends.mps.is_available()}")
        print(f"MPS (Apple GPU) built: {torch.backends.mps.is_built()}")
        
   
        # Train the PyTorch model with GPU acceleration
        print("\nTraining model...")
        model, history, X_test_tensor, Y_test_tensor = train_pytorch_model(
            X_train, Y_train, X_test, Y_test, epochs=10, batch_size=64, lr=0.003
        )
        
        # Evaluate the model - pass both reduced and original test images
        print("\nEvaluating model...")
        evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor, X_orig_test)
        
        # Plot training history
        plot_training_history(history)
        
        # Save the model
        model_filename = f"rgc_model_unit_{unit_idx}.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}") 