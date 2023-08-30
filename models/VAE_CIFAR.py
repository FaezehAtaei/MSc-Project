from __future__ import print_function

import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F


# defining VAE model for CIFAR-10 without memory
class VAE_CIFAR(nn.Module):
    def __init__(self):
        super(VAE_CIFAR, self).__init__()
        
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
        

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(256, 512)
        self.fc_mu = nn.Linear(512, 32)
        self.fc_log_var = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 256)


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
        h1 = F.relu(self.dropout(self.enc2(h0)))
        h2 = F.relu(self.enc3(h1))
        h3 = F.relu(self.dropout(self.enc4(h2)))

        batch, _, _, _ = h3.shape
        h3 = F.adaptive_avg_pool2d(h3, 1).reshape(batch, -1)
        h4 = F.relu(self.fc1(h3))
        return self.fc_mu(h4), self.fc_log_var(h4)

    # Perform reparameterization trick to sample from the latent space
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Decode the latent representation tensor z through the decoder layers with ReLU + Sigmoid activation
    def decode(self, z):
        h0 = F.relu(self.dec1(z))
        h1 = F.relu(self.dec2(h0))
        h2 = F.relu(self.dropout(self.dec3(h1)))
        return torch.sigmoid(self.dec4(h2))

    # Perform the full forward pass of the autoencoder
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = F.relu(self.fc2(z))
        z = z.view(-1, 256, 1, 1)
        return self.decode(z), mu, logvar

