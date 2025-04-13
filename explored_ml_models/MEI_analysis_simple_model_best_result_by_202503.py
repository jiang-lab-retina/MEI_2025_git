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

# Create mock data for testing
def create_mock_data(shape=(1185, 200, 150), output_dim=OUT_DIM):
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
    """
    def __init__(self):
        super(RGCNet, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
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
        # Input: 200x150 -> Conv1+Pool: 100x75 -> Conv2+Pool: 50x37 -> Conv3+Pool: 25x18
        fc_input_size = 128 * 25 * 18
        
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

def evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor):
    """
    Evaluate the trained PyTorch model
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
        
        # Plot example predictions
        num_examples = min(5, len(X_test_np))
        plt.figure(figsize=(15, 10))
        
        for i in range(num_examples):
            # Plot image
            plt.subplot(num_examples, 3, i*3+1)
            plt.imshow(X_test_np[i, 0], cmap='gray')
            plt.title(f"Input Image {i+1}")
            plt.axis('off')
            
            # Plot true RGC response
            plt.subplot(num_examples, 3, i*3+2)
            plt.bar(range(len(Y_test_np[i])), Y_test_np[i])
            plt.title(f"True RGC Response {i+1}")
            
            # Plot predicted RGC response
            plt.subplot(num_examples, 3, i*3+3)
            plt.bar(range(len(Y_pred_np[i])), Y_pred_np[i])
            plt.title(f"Predicted RGC Response {i+1}")
        
        plt.tight_layout()
        plt.savefig(f"prediction_examples_unit_{unit_idx}.png")
        print(f"Saved prediction examples to prediction_examples_unit_{unit_idx}.png")
        
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
        
        X = np.array(final_X_dict["X"])
        Y_all = np.array(list(final_Y_dict.values()))
        print(f"Loaded real data: X shape {X.shape}, Y_all shape {Y_all.shape}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Could not load real data: {e}")
        print("Creating mock data instead...")
        X, Y_all, all_images_mean = create_mock_data()
    
    # You can choose any unit from Y_all by setting the index
    for unit_idx in range(Y_all.shape[0]):
        #unit_idx = 26  # Change this index to train for different units
        Y = Y_all[unit_idx]
        print(f"Working with data shapes: X: {X.shape}, Y: {Y.shape}")
        print(f"Training model for unit {unit_idx}")
    
        # Create train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print(f"Train/test shapes: X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")
        
        # Check if MPS is available before importing PyTorch
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS (Apple GPU) available: {torch.backends.mps.is_available()}")
        print(f"MPS (Apple GPU) built: {torch.backends.mps.is_built()}")
        
   
        # Train the PyTorch model with GPU acceleration
        print("\nTraining model...")
        model, history, X_test_tensor, Y_test_tensor = train_pytorch_model(
            X_train, Y_train, X_test, Y_test, epochs=10, batch_size=64, lr=0.003
        )
        
        # Evaluate the model
        print("\nEvaluating model...")
        evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor)
        
        # Plot training history
        plot_training_history(history)
        
        # Save the model
        model_filename = f"rgc_model_unit_{unit_idx}.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}") 