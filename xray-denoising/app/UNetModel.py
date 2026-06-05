import numpy as np
import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # encoder layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(True)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = nn.Conv2d(32 + 16, 16, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU(True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = nn.Conv2d(16 + 1, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        - Input size: torch.Size([b, 1, 512, 736])
        - Conv1 output size: torch.Size([b, 16, 256, 368])
        - Conv2 output size: torch.Size([b, 32, 128, 184])
        - Upsample3 output size: torch.Size([b, 32, 256, 368])
        - Concatenated size after upsample3: torch.Size([b, 48, 256, 368])
        - Conv7 output size: torch.Size([b, 16, 256, 368])
        - Upsample4 output size: torch.Size([b, 16, 512, 736])
        - Concatenated size after upsample4: torch.Size([b, 17, 512, 736])
        - Conv8 output size: torch.Size([b, 1, 512, 736])
        - Final output size: torch.Size([b, 1, 512, 736])
        """
        
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x2 = self.conv2(x1)
        x2 = self.relu2(x2)
        
        x3 = self.upsample3(x2)
        x3 = torch.cat([x3, x1], dim=1)
        x4 = self.conv7(x3)
        x4 = self.relu7(x4)
        x5 = self.upsample4(x4)
        x5 = torch.cat([x5, x], dim=1)
        x6 = self.conv8(x5)
        x6 = self.sigmoid(x6)
        
        return x6