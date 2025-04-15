import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1 channel (grayscale), Output: 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Input: 32 channels, Output: 64 channels
        self.pool = nn.MaxPool2d(2, 2) # 2x2 max pooling
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Input: 64 channels * 7 * 7, Output: 128 features
        self.fc2 = nn.Linear(128, 10)   # Input: 128 features, Output: 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Convolution -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv2(x))) # Convolution -> ReLU -> Pooling
        x = x.view(-1, 64 * 7 * 7)           # Flatten the feature map
        x = F.relu(self.fc1(x))             # Fully connected layer -> ReLU
        x = self.fc2(x)                      # Fully connected layer (output)
        return x
