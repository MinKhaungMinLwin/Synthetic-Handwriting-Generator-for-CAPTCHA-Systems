"""
visualize.py
------------
Utility functions for displaying batches of images.
"""

import matplotlib.pyplot as plt
import torch
import math

def show_batch(images, labels, n=16):
    """
    Visualizes a batch of MNIST images with labels.
    """
    n = min(n, len(images), len(labels))
    if n < 1:
        raise ValueError("at least one image is required")
    columns = math.ceil(math.sqrt(n))
    rows = math.ceil(n / columns)
    
    fig, axes = plt.subplots(rows, columns, figsize=(1.8 * columns, 1.8 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(images[i].detach().squeeze().cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return fig


def plot_real_cgan_diffusion(real_imgs, cgan_imgs, diffusion_imgs, n_classes=10):
    """Plot one real, cGAN, and diffusion image for each class."""
    rows = (real_imgs, cgan_imgs, diffusion_imgs)
    titles = ("Real", "cGAN", "Diffusion")
    fig, axes = plt.subplots(3, n_classes, figsize=(1.4 * n_classes, 4.5), squeeze=False)
    for row, (images, title) in enumerate(zip(rows, titles)):
        for label in range(n_classes):
            axes[row, label].imshow(images[label].detach().squeeze().cpu(), cmap="gray", vmin=-1, vmax=1)
            axes[row, label].axis("off")
            if row == 0:
                axes[row, label].set_title(str(label))
        axes[row, 0].set_ylabel(title)
    plt.tight_layout()
    plt.show()
    return fig
