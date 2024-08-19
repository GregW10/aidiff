import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=3, output_size=(256, 256)):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 4096)
        self.fc4 = nn.Linear(4096, 16 * 16 * 64)
        
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.upsample4 = nn.Upsample(size=output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        
        x = x.view(-1, 64, 16, 16)
        
        x = self.relu(self.conv1(x))
        x = self.upsample1(x)
        
        x = self.relu(self.conv2(x))
        x = self.upsample2(x)
        
        x = self.relu(self.conv3(x))
        x = self.upsample3(x)

        x = self.conv4(x)
        x = self.upsample4(x)
        
        return x
