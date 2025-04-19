import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim  # Import optimizer

import time
from tqdm import tqdm 
import VQ_VAE_2_1D_data 
from VQ_VAE_2_1D_data import create_dataloader
from scipy.io.wavfile import write  # For saving audio
class ResidualBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0) # 1x1 equivalent


        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        #print(f"ResidualBlock1D input shape: {x.shape}")
        residual = self.skip(x)
        out = F.relu(x)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = out + residual
        #print(f"ResidualBlock1D output shape: {out.shape}")
        return out



