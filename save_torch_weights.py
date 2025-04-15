import torch
import numpy as np
from model import Net  # Your PyTorch model definition

# Load the PyTorch model and weights
torch_model = Net()
PATH = './mnist_cnn.pth'
state_dict = torch.load(PATH)
torch_model.load_state_dict(state_dict)
torch_model.eval()
state_dict = torch_model.state_dict()

# Convert state_dict tensors to numpy arrays, transposing Conv2d weights
weights_np = {}
for k, v in state_dict.items():
    if "conv" in k and "weight" in k:
        # Transpose from (C_out, C_in, H, W) to (C_out, H, W, C_in)
        weights_np[k] = v.numpy().transpose(0, 2, 3, 1)
        print(f"Transposed {k} from {v.shape} to {weights_np[k].shape}")
    else:
        # Keep Linear weights/biases and Conv biases as is
        weights_np[k] = v.numpy()

# Define the output npz file name
NPZ_PATH = './mnist_weights_transposed.npz' # Use a new name

# Save the potentially transposed numpy weights to an npz file
np.savez(NPZ_PATH, **weights_np)

print(f"PyTorch weights (with conv transposed) saved to {NPZ_PATH}")
print("Saved keys:", list(weights_np.keys()))
