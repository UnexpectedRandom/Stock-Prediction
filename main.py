import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read the CSV file
file = pd.read_csv('AAPL.csv')

# Extract the 'High' column data
column_name = 'High'
column_data = file[column_name]

# Convert the column data to a PyTorch tensor
tensor_data = torch.tensor(column_data.values, dtype=torch.float32)

# Manual split into training and testing sets
train_size = int(0.8 * len(tensor_data))  # 80% for training
X_train, X_test = tensor_data[:train_size], tensor_data[train_size:]
y_train, y_test = tensor_data[:train_size], tensor_data[train_size:]

# Define the neural network model
class HighsAlgorithm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = HighsAlgorithm().to(device)

epochs = 1000

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = torch.nn.L1Loss()

# Training loop
for epoch in range(epochs):
    model.train()

    # Forward pass
    y_pred = model(X_train.unsqueeze(1))  # Unsqueeze to add batch dimension
    
    # Compute loss
    loss = loss_fn(y_pred, y_train.unsqueeze(1))  # Unsqueeze to match shape
    
    # Backpropagation
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    # Evaluation on testing set
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test.unsqueeze(1))  # Unsqueeze to add batch dimension
        test_loss = loss_fn(test_pred, y_test.unsqueeze(1))  # Unsqueeze to match shape
        
    # Print training and testing loss for each epoch (optional)
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
