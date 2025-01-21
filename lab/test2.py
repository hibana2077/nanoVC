import time
import torch
import torch.nn as nn
import torch.nn.functional as F


transconv1 = nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)

x = torch.rand(1, 256, 2768)
x = transconv1(x)
print(x.shape)  # torch.Size([1, 128, 5536])