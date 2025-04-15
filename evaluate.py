import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net  # Import the model from model.py

# Define transformations (same as during training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the test dataset
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Create an instance of the network
net = Net()

# Load the trained model's weights
PATH = './mnist_cnn.pth'  # Path to the saved model
net.load_state_dict(torch.load(PATH))  #Load the weights into the model

# Set the model to evaluation mode
net.eval() #Important: disables training-specific layers such as dropout

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation during evaluation
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1) #Get the index of the max log-probability
        total += labels.size(0) #Increment the total number of samples
        correct += (predicted == labels).sum().item() #Increment the correct count

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
