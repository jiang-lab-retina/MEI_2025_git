import sys
from os import path
import os
#sys.path.insert(1, '../')

#load the path to the directory containing the package

#sys.path.append(path.dirname(path.dirname(os.getcwd())))
#import jianglab as jl
#import McsPy.McsCMOSMEA as McsCMOSMEA
#from load_raw_data import load_basic_cmtr_data
#from feature_analysis import *
import matplotlib.pyplot as plt
from MEI_analysis_helper import *
import glob
import pickle
import pandas as pd
from tqdm import tqdm

#### create model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.model_selection import train_test_split


# def load_dataset(image_firing_csv_path, 
#                  receptive_field_pkl_path,
#                  compressed_image_data_path_list,
#                  ):
#     all_X_Y_dict = {}
#     df = pd.read_csv(image_firing_csv_path)
#     df.set_index("Unnamed: 0", inplace=True)
#     receptive_field_dict = pickle.load(open(receptive_field_pkl_path, "rb"))
#     compressed_image_data = []
#     image_set_lengnth_list = []
#     for compressed_image_data_path in compressed_image_data_path_list:
#         compressed_image_data.extend(np.load(compressed_image_data_path))
#         image_set_lengnth_list.append(np.load(compressed_image_data_path).shape[0])
#     compressed_image_data = np.array(compressed_image_data)
#     print(compressed_image_data.shape, df.shape)
    
#     for row_index, row in df.iterrows():
#         X, Y = [], []
#         receptive_field_center = receptive_field_dict[row_index-1]
#         receptive_field_radius = 200
#         receptive_field_mask = np.zeros((200, 150))
#         #avoid the edge effect by avoiding the negative index
#         receptive_field_mask[max(0, receptive_field_center[0] - receptive_field_radius):min(200, receptive_field_center[0] + receptive_field_radius),
#                             max(0, receptive_field_center[1] - receptive_field_radius):min(150, receptive_field_center[1] + receptive_field_radius)] = 1
        
#         for i, (col_index, col) in enumerate(row.iteritems()):
#             #image_index = int(col_index.split("--")[-1])
#             X.append(compressed_image_data[i, :, :] )
#             Y.append(eval(col))
#         X = np.array(X)
#         Y = np.array(Y)
#         print(X.shape, Y.shape, row_index)
#         all_X_Y_dict[row_index] = (X, Y)
#     return all_X_Y_dict

# def visualize_X_Y_by_star(all_X_Y_dict, all_images_mean):
#     row_num = 12
#     col_num = len(all_X_Y_dict) // row_num + 1
#     fig, axs = plt.subplots(row_num, col_num)
#     for row_index, (X, Y) in all_X_Y_dict.items():
#         Y_max_list = np.max(Y, axis=1)
#         sta_image_all = []
#         for image, y in zip(X, Y_max_list):
#             weighted_image = image * y
#             sta_image_all.append(weighted_image)
#         unit_sta_image = np.mean(sta_image_all, axis=0) - (all_images_mean * np.mean(Y_max_list))
#         min_value = np.min(unit_sta_image[unit_sta_image !=0 ])
#         max_value = np.max(unit_sta_image[unit_sta_image !=0])
#         axs[(row_index-1) // col_num, (row_index-1) % col_num].imshow(unit_sta_image, cmap="jet", vmin=min_value, vmax=max_value)
#     plt.show()

def load_final_X_Y_dict(final_X_Y_dict_path):
    with open(final_X_Y_dict_path, "rb") as f:
        final_X_Y_dict = pickle.load(f)
    return final_X_Y_dict

def visualize_final_X_Y_dict(final_X_dict, final_Y_dict, all_images_mean):
    row_num = 12
    col_num = len(final_Y_dict) // row_num + 1
    fig, axs = plt.subplots(row_num, col_num)
    for i, (unit_id, Y) in enumerate(tqdm(final_Y_dict.items())):
        X = final_X_dict["X"]
        Y_max_list = np.max(Y, axis=1)
        sta_image_all = []
        for image, y in zip(X, Y_max_list):
            weighted_image = image * (y/200)
            sta_image_all.append(weighted_image)
        unit_sta_image = np.mean(sta_image_all, axis=0) - (all_images_mean * np.mean(Y_max_list)/200)
        axs[i//col_num, i%col_num].imshow(unit_sta_image, cmap="jet")
    plt.show()
    
    

    
# Define the CNN model
class RGCNet(nn.Module):
    """
    A 3-layer CNN for predicting RGC outputs from visual stimulation.
    
    Architecture:
    - Input: (batch, 1, 200, 150) grayscale images
    - Conv1: 32 filters, 7x7 kernel, ReLU activation, batch normalization
    - MaxPool1: 2x2 pooling
    - Conv2: 64 filters, 5x5 kernel, ReLU activation, batch normalization
    - MaxPool2: 2x2 pooling
    - Conv3: 128 filters, 3x3 kernel, ReLU activation, batch normalization
    - MaxPool3: 2x2 pooling
    - Fully connected layers: 256 -> 128 -> 39 (output)
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
            nn.Linear(128, 39)  # Output size for RGC prediction
        )
        
    def forward(self, x):
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply fully connected layers
        x = self.fc(x)
        
        return x

def train_pytorch_model(X_train, Y_train, X_test, Y_test, epochs=50, batch_size=32, lr=0.001):
    """
    Train a PyTorch model for RGC output prediction
    
    Args:
        X_train: Training images, shape (samples, height, width)
        Y_train: Training RGC outputs, shape (samples, output_size)
        X_test: Test images, shape (samples, height, width)
        Y_test: Test RGC outputs, shape (samples, output_size)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    # Convert to PyTorch tensors
    # Add channel dimension and normalize
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RGCNet().to(device)
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
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
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

def evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor):
    """
    Evaluate the trained PyTorch model
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        Y_test_tensor = Y_test_tensor.to(device)
        Y_pred = model(X_test_tensor)
        
        # Calculate metrics
        mse = nn.MSELoss()(Y_pred, Y_test_tensor).item()
        mae = torch.mean(torch.abs(Y_pred - Y_test_tensor)).item()
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Convert to numpy for plotting
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
    plt.show()
    
    return mse, mae

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
    plt.show()
    
    
    
if __name__ == "__main__":
    image_firing_csv_path = "2025_connected_image_to_firing_time.csv"
    compressed_image_data_path_list = glob.glob("compressed_ILSVRC2012/*.npy")
    compressed_image_data_path_list.sort()
    # all_X_Y_dict = load_dataset(image_firing_csv_path = image_firing_csv_path,
    #                             receptive_field_pkl_path = "receptive_field_dict.pkl",
    #                             compressed_image_data_path_list = compressed_image_data_path_list,
    #                             )
    # print(list(all_X_Y_dict.values())[0][0].shape, list(all_X_Y_dict.values())[0][1].shape)
    # all_images_mean = np.load("all_images_mean.npy")
    # visualize_X_Y_by_star(all_X_Y_dict, all_images_mean)
    all_images_mean = np.load("all_images_mean.npy")
    final_X_dict = load_final_X_Y_dict("final_X_dict.pkl")
    final_Y_dict = load_final_X_Y_dict("final_Y_dict.pkl")
    
    
    # visualize_final_X_Y_dict(final_X_dict, final_Y_dict, all_images_mean)
    
    X = np.array(final_X_dict["X"])
    Y_all = np.array(list(final_Y_dict.values()))
    print(X.shape, Y_all.shape)
    
    cell_index = 26
    Y = Y_all[cell_index]
    X = X       
    print(X.shape, Y.shape)
   
    #### create train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    
    
    
    # Train the PyTorch model with 5 epochs for testing
    model, history, X_test_tensor, Y_test_tensor = train_pytorch_model(
        X_train, Y_train, X_test, Y_test, epochs=20, batch_size=32, lr=0.001
    )
    
    # Evaluate the model
    evaluate_pytorch_model(model, X_test_tensor, Y_test_tensor)
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    torch.save(model.state_dict(), f"rgc_model_unit_{cell_index}.pt")
    print(f"Model saved to rgc_model_unit_{cell_index}.pt")
    

