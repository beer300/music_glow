import os
import os.path
import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
from data_load import WavDataset  # Import WavDataset from data_load.py
from data_load import spectrum2wav  # Import spectrum2wav from data_load.py
from tqdm import tqdm  # Import tqdm for the progress bar
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        skip = self.skip_conv(x)
        x = self.relu(x + residual)
        return x + skip, skip
class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(x)
class Wavenet(nn.Module):
    def __init__(self,):
        super(Wavenet, self).__init__()
    
    def forward(self, x):
        
        return x