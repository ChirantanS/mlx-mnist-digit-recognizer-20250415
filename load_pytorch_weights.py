import torch
from model import Net  # Import your PyTorch model definition

# Create an instance of the network
net = Net()

# Define the path to the saved model weights
PATH = './mnist_cnn.pth'

# Load the model's state dictionary
net.load_state_dict(torch.load(PATH))

# Print the state dictionary (optional, for inspection)
for name, param in net.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

print("PyTorch model weights loaded successfully!")