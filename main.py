import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn

torch.manual_seed(42)
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

X_train = X_train.unsqueeze(1)
y_train = y_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)
y_test = y_test.unsqueeze(1)


# Define the simpler neural network model
class SimpleHighsAlgorithm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=2)  # Simplified number of output features
        self.relu = nn.ReLU()  # Or replace with simpler activation like nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=2, out_features=1)  # Simplified number of input features

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the simpler model
simple_model = SimpleHighsAlgorithm().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)  # You might need to adjust the learning rate

# Training loop
epochs = 1000
for epoch in range(epochs):
    simple_model.train()
    
    optimizer.zero_grad()
    
    outputs = simple_model(X_train)
    
    loss = criterion(outputs, y_train)
    
    loss.backward()
    
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}')

# After training, you can use the model to make predictions on the test set
simple_model.eval()

with torch.no_grad():
    test_outputs = simple_model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')


tensor_data = tensor_data.unsqueeze(1)

# Move tensor_data to the appropriate device
tensor_data = tensor_data.to(device)


# Make predictions using the trained model
with torch.no_grad():
    predictions = simple_model(tensor_data)

# Convert predictions tensor to numpy array
predictions = predictions.cpu().numpy()

# Convert tensor_data to numpy array
tensor_data_np = tensor_data.cpu().numpy()

# Plot original data as blue lines
plt.plot(range(len(tensor_data_np)), tensor_data_np, label='Original Data', color='blue')

# Plot predictions as orange dots
plt.scatter(range(len(predictions)), predictions, label='Predictions', color='orange')

plt.xlabel('Data Point Index')
plt.ylabel('High Value')
plt.title('Original Data vs. Predictions')
plt.legend()
plt.show()

