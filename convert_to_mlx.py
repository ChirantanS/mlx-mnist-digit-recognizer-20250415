import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from torchvision import datasets, transforms
from model import Net as PyTorchNet # Rename import

# Define the equivalent MLX model (using NHWC logic implied by load_weights)
class MLXNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer definition implies C_in=1, C_out=32 for conv1 etc.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Based on NHWC pooling from (1,28,28,1) -> (1,7,7,64) -> flattened 3136
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x): # Input x must be NHWC (1, 28, 28, 1)
        # Use print statements for debugging if needed
        # print(f"  MLX __call__ input shape: {x.shape}")
        x = self.conv1(x) # nn.Conv2d likely adapts to NHWC weights
        x = mx.maximum(x, 0) # ReLU
        # print(f"  MLX after conv1+relu shape: {x.shape}")
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # print(f"  MLX after pool1 shape: {x.shape}")
        x = self.conv2(x)
        x = mx.maximum(x, 0) # ReLU
        # print(f"  MLX after conv2+relu shape: {x.shape}")
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # print(f"  MLX after pool2 shape: {x.shape}")
        x = x.reshape(x.shape[0], -1) # Flatten NHWC
        # print(f"  MLX after reshape shape: {x.shape}")
        x = self.fc1(x)
        x = mx.maximum(x, 0) # ReLU
        # print(f"  MLX after fc1+relu shape: {x.shape}")
        return self.fc2(x)

# Instantiate the MLX model
mlx_model = MLXNet()

# Load weights from the CORRECTLY TRANSPOSED npz file
NPZ_PATH = './mnist_weights_transposed.npz' # Use the file with transposed conv weights
mlx_model.load_weights(NPZ_PATH)

print(f"Loaded MLX weights from {NPZ_PATH} using model.load_weights()")
mx.eval(mlx_model.parameters())

# --- Comparison Section ---
# Load original PyTorch model just for comparison output
torch_model_orig = PyTorchNet()
torch_model_orig.load_state_dict(torch.load('./mnist_cnn.pth'))
torch_model_orig.eval()

# Load sample data (normalized)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
sample_idx = 0
sample_image, sample_label = testset[sample_idx] # Shape (1, 28, 28), normalized

# PyTorch prediction (Input NCHW: 1, 1, 28, 28)
with torch.no_grad():
    torch_output = torch_model_orig(sample_image.unsqueeze(0))
    torch_prediction = torch.argmax(torch_output).item()

# MLX prediction (Input NHWC: 1, 28, 28, 1)
sample_image_np = sample_image.numpy() # Shape (1, 28, 28), normalized

# Convert to MLX array
mlx_image = mx.array(sample_image_np) # Shape (1, 28, 28)

# Reshape/Transpose to NHWC: (batch_size, height, width, channels) = (1, 28, 28, 1)
mlx_image = mx.expand_dims(mlx_image, axis=3) # Add channel dim at the end -> (1, 28, 28, 1)

print(f"MLX Input Image Shape before model call: {mlx_image.shape}")

# Perform inference with MLX model
mlx_output = mlx_model(mlx_image)
mx.eval(mlx_output)
mlx_prediction = int(np.argmax(np.array(mlx_output)))

print(f"Sample image label: {sample_label}")
print(f"PyTorch prediction: {torch_prediction}")
print(f"MLX prediction: {mlx_prediction}")

if torch_prediction == mlx_prediction:
    print("SUCCESS: PyTorch and MLX predictions match!")
else:
    print("ERROR: PyTorch and MLX predictions DO NOT match!")

print("Conversion using transposed NPZ and model.load_weights complete!")
