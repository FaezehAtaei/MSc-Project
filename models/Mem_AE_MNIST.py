from __future__ import print_function

import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F
from models.memory_module import MemModule


# defining memory augmented autoencoder model for MNIST
class Mem_AE_MNIST(nn.Module):
    def __init__(self):
        super(Mem_AE_MNIST, self).__init__()
        
        # Define a dropout layer with a dropout probability of 0.2
        self.dropout = nn.Dropout2d(p=0.2)

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3,
            stride=2, padding=0)

        self.enc2 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3,
            stride=2, padding=1)

        self.enc3 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3,
            stride=2, padding=1)

        
        # memory
        self.mem_rep = MemModule(mem_dim=100, fea_dim=8, shrink_thres=0.0025)
        

        # decoder
        self.dec2 = nn.ConvTranspose2d(
            in_channels=8, out_channels=16, kernel_size=4,
            stride=1, padding=0)

        self.dec3 = nn.ConvTranspose2d(
            in_channels=16, out_channels=32, kernel_size=2,
            stride=2, padding=0)

        self.dec4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=2,
            stride=2, padding=0)

    # Encode the input tensor x through the encoder layers with ReLU activation    
    def encode(self, x):
        h0 = F.relu(self.enc1(x))
        h1 = F.relu(self.enc2(h0))
        h2 = F.relu(self.enc3(h1))
        return h2

    # Decode the latent representation tensor z through the decoder layers with ReLU + Sigmoid activation
    def decode(self, z):
        h1 = F.relu(self.dec2(z))
        h2 = F.relu(self.dec3(h1))
        return torch.sigmoid(self.dec4(h2))

    # Perform the full forward pass of the autoencoder
    def forward(self, x):
        f = self.encode(x)
        res_mem = self.mem_rep(f)
        z = res_mem['output']
        att = res_mem['att']
        return self.decode(z), att
