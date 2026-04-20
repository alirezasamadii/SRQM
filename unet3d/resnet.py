import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual block used in ResNet can be retained for feature extraction
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

# ResNet-like Super-Resolution model
class ResNet50(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=5, upscale_factor=2):
        super(ResNet50, self).__init__()  # Class name and super() should match
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)
        
        # Stack residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        # Upsampling using transposed convolution
        #self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0)
        #self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)  # suggestion : padding should be equal to upscaling

        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=9, padding=4)

    def forward(self, x):

        #input_first_three_channels = x[:, :3, :, :]
        x = self.relu(self.conv1(x))
        x = self.residual_blocks(x)
        #x = self.upsample(x)  # Upsample to higher resolution
        x = self.conv_out(x)
        
        # Apply ReLU + epsilon for final activation
      
        #x = F.relu(x) + input_first_three_channels
        x = F.relu(x)
        #max_vals = torch.tensor([1.6, 4, 1], device=x.device).view(1, -1, 1, 1)  # Shape: [1, C, 1, 1]
        #x = torch.min(x, max_vals)
        return x
