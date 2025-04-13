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
    X = np.random.randint(0, 256, size=shape).astype(np.uint8)
    Y_all = np.random.uniform(0, 10, size=(1, shape[0], output_dim)).astype(np.float32)
    all_images_mean = np.mean(X, axis=0)
    print(f"Created mock X data with shape: {X.shape}")
    print(f"Created mock Y data with shape: {Y_all.shape}")
    return X, Y_all, all_images_mean

# Updated CNN model with Global Average Pooling and parameterized dropout rate
class RGCNet(nn.Module):
    """
    A 3-layer CNN for predicting RGC outputs from visual stimulation,
    using global average pooling to reduce parameters.
    """
    def __init__(self, dropout_rate=0.3):
        super(RGCNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 200x150 -> 100x75
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 100x75 -> 50x37
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 50x37 -> 25x18 (but we won't use this spatial size directly)
        )
        # Global Average Pooling instead of flattening the entire spatial map
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # output shape: (batch, 128, 1, 1)
        
        # Fully connected layers: input is now 128 after GAP
        self.fc = nn.Sequential(
            nn.Flatten(),  # (batch, 128)
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, OUT_DIM)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = self.fc(x)
        return x

# Update device selection
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
    Train a PyTorch CNN model for RGC output prediction.
    Hyperparameters (lr, batch_size, dropout_rate) are passed as arguments.
    """
    try:
        device = get_device()
        # Convert to PyTorch tensors, add channel dimension and normalize
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
        
        print(f"Tensor shapes: X_train: {X_train_tensor.shape}, Y_train: {Y_train_tensor.shape}")
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model with specified dropout_rate
        model = RGCNet(dropout_rate=dropout_rate).to(device)
        verify_gpu_usage(next(model.parameters()), "Model")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
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
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
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
        
        X_test_np = X_test_tensor.cpu().numpy()
        Y_test_np = Y_test_tensor.cpu().numpy()
        Y_pred_np = Y_pred.cpu().numpy()
        
        num_examples = min(5, len(X_test_np))
        plt.figure(figsize=(15, 10))
        for i in range(num_examples):
            plt.subplot(num_examples, 3, i*3+1)
            plt.imshow(X_test_np[i, 0], cmap='gray')
            plt.title(f"Input Image {i+1}")
            plt.axis('off')
            plt.subplot(num_examples, 3, i*3+2)
            plt.bar(range(len(Y_test_np[i])), Y_test_np[i])
            plt.title(f"True RGC Response {i+1}")
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

def plot_training_history(history, unit_idx):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
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

# Hyperparameter tuning for unit index 20 (the 21st cell)
def hyperparameter_tuning(X_train, Y_train, X_val, Y_val, param_grid, epochs=10):
    best_loss = float('inf')
    best_params = None
    best_history = None
    for lr in param_grid['lr']:
        for batch_size in param_grid['batch_size']:
            for dropout_rate in param_grid['dropout_rate']:
                print(f"Testing configuration: lr={lr}, batch_size={batch_size}, dropout_rate={dropout_rate}")
                model, history, _, _ = train_pytorch_model(
                    X_train, Y_train, X_val, Y_val,
                    epochs=epochs, batch_size=batch_size, lr=lr, dropout_rate=dropout_rate
                )
                val_loss = history['val_loss'][-1]
                print(f"Configuration lr={lr}, batch_size={batch_size}, dropout_rate={dropout_rate} yielded val_loss={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = {'lr': lr, 'batch_size': batch_size, 'dropout_rate': dropout_rate}
                    best_history = history
    print(f"Best hyperparameters found: {best_params} with validation loss: {best_loss:.4f}")
    return best_params, best_history

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
        Y_all = np.array(list(final_Y_dict.values()))
        print(f"Loaded real data: X shape {X.shape}, Y_all shape {Y_all.shape}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Could not load real data: {e}")
        print("Creating mock data instead...")
        X, Y_all, all_images_mean = create_mock_data()
    
    # Loop through each unit; use hyperparameter tuning for unit index 20 (21st cell)
    for unit_idx in range(Y_all.shape[0]):
        Y = Y_all[unit_idx]
        print(f"Working with data shapes: X: {X.shape}, Y: {Y.shape}")
        print(f"Processing model for unit {unit_idx}")
    
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print(f"Train/test shapes: X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")
    
        if unit_idx == 20:
            # Hyperparameter tuning for the 21st cell
            param_grid = {
                'lr': [0.001, 0.003, 0.01],
                'batch_size': [16, 32, 64],
                'dropout_rate': [0.2, 0.3, 0.5]
            }
            best_params, tuning_history = hyperparameter_tuning(X_train, Y_train, X_test, Y_test, param_grid, epochs=10)
            # Use best hyperparameters to train final model
            model, history, X_test_tensor, Y_test_tensor = train_pytorch_model(
                X_train, Y_train, X_test, Y_test,
                epochs=10,
                batch_size=best_params['batch_size'],
                lr=best_params['lr'],
                dropout_rate=best_params['dropout_rate']
            )
        else:
            # Use default hyperparameters for other cells
            model, history, X_test_tensor, Y_test_tensor = train_pytorch_model(
                X_train, Y_train, X_test, Y_test, epochs=10, batch_size=64, lr=0.003, dropout_rate=0.3
            )
    
        evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor, unit_idx)
        plot_training_history(history, unit_idx)
    
        model_filename = f"rgc_model_unit_{unit_idx}.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")