from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


# defining autoencoder model for CIFAR-10 without memory
class AE_CIFAR(nn.Module):
    def __init__(self):
        super(AE_CIFAR, self).__init__()
        
        # Define a dropout layer with a dropout probability of 0.2
        self.dropout = nn.Dropout2d(p=0.2)

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3,
            stride=2, padding=1)

        self.enc2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3,
            stride=2, padding=1)

        self.enc3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3,
            stride=2, padding=1)

        self.enc4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3,
            stride=2, padding=0
        )

        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4,
            stride=2, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=4,
            stride=2, padding=1)

        self.dec3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4,
            stride=2, padding=1)

        self.dec4 = nn.ConvTranspose2d(
            in_channels=64, out_channels=3, kernel_size=4,
            stride=2, padding=1)
       
    # Encode the input tensor x through the encoder layers with ReLU activation
    def encode(self, x):
        h0 = F.relu(self.enc1(x))
        h1 = F.relu(self.enc2(h0))
        h2 = F.relu(self.enc3(h1))
        h3 = F.relu(self.enc4(h2))
        return h3
    
    # Decode the latent representation tensor z through the decoder layers with ReLU + Sigmoid activation
    def decode(self, z):
        h0 = F.relu(self.dec1(z))
        h1 = F.relu(self.dec2(h0))
        h2 = F.relu(self.dec3(h1))
        return torch.sigmoid(self.dec4(h2))

    # Perform the full forward pass of the autoencoder
    def forward(self, x):
        f = self.encode(x)
        return self.decode(f)

