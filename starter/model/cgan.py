"""
cgan.py
-------
Defines Generator and Discriminator architectures for Conditional GAN.
"""

import torch
import torch.nn as nn
import numpy as np


def _validate_labels(labels, batch_size):
    """Return class labels in the format expected by ``nn.Embedding``."""
    if labels.ndim != 1 or labels.size(0) != batch_size:
        raise ValueError(f"labels must have shape ({batch_size},), got {tuple(labels.shape)}")
    return labels.long()

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.z_dim = z_dim
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = 7

        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, 128 * self.init_size ** 2),
            nn.BatchNorm1d(128 * self.init_size ** 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (128, self.init_size, self.init_size)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, img_shape[0], 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        if noise.ndim != 2 or noise.size(1) != self.z_dim:
            raise ValueError(
                f"noise must have shape (batch, {self.z_dim}), got {tuple(noise.shape)}"
            )
        labels = _validate_labels(labels, noise.size(0))
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(x)
        return img.reshape(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(num_classes, int(np.prod(img_shape)))
        
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0] + 1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        labels = _validate_labels(labels, img.size(0))
        label_map = self.label_emb(labels).reshape(
            img.size(0), 1, self.img_shape[1], self.img_shape[2]
        )
        x = torch.cat((img, label_map), dim=1)
        validity = self.model(x)
        return validity
