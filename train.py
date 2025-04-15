import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net  # Import the model from model.py

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the training and testing datasets
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Create an instance of the network
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  #Loss function for classification
optimizer = optim.Adam(net.parameters(), lr=0.001) #Adam optimzer

# Training loop
num_epochs = 2  # Adjust the number of epochs as needed (start with a small number)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = net(inputs)  # Forward pass
        loss = criterion(outputs, labels) #Calculate the loss
        loss.backward() #Backpropagation to calculate gradients
        optimizer.step()  #Update the weights

        running_loss += loss.item()
        if i % 200 == 199:    # Print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# --- Optional: Save the trained model ---
PATH = './mnist_cnn.pth' #Path to save the model
torch.save(net.state_dict(), PATH) #save the model's weights
print(f'Model saved to {PATH}')
