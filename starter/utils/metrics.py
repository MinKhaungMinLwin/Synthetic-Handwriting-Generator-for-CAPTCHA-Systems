"""Evaluation helpers for generative quality and downstream utility."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _fid_images(images):
    """Convert normalized MNIST BCHW images to RGB images in [0, 1]."""
    if images.ndim != 4 or images.size(1) not in (1, 3):
        raise ValueError("images must have shape (N, 1|3, H, W)")
    images = images.float()
    if images.min() < 0:
        images = (images + 1) / 2
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    return images.clamp(0, 1)


@torch.no_grad()
def compute_fid(real_images, fake_images, device="cpu", batch_size=128):
    """Compute FID between real and synthetic tensors using TorchMetrics."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise ImportError(
            "compute_fid requires torchmetrics and torch-fidelity"
        ) from exc

    requested_device = torch.device(device)
    # TorchMetrics stores FID covariance statistics as float64. Apple MPS does
    # not support float64 tensors, so feature extraction/accumulation must run
    # on CPU while model generation can remain accelerated on MPS.
    metric_device = torch.device("cpu") if requested_device.type == "mps" else requested_device
    metric = FrechetInceptionDistance(normalize=True).to(metric_device)
    for images, is_real in ((real_images, True), (fake_images, False)):
        images = _fid_images(images)
        for start in range(0, len(images), batch_size):
            metric.update(
                images[start:start + batch_size].to(metric_device), real=is_real
            )
    return float(metric.compute().cpu())


class SimpleCNN(nn.Module):
    """Small MNIST classifier used for downstream synthetic-data evaluation."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def train_classifier_on_synthetic(
    images, labels, device="cpu", epochs=5, batch_size=128, lr=1e-3
):
    """Train and return a classifier using labeled synthetic images."""
    model = SimpleCNN().to(device)
    loader = DataLoader(TensorDataset(images, labels.long()), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_images.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def classifier_accuracy(model, dataloader, device="cpu"):
    """Return classification accuracy for a dataloader."""
    model.eval()
    correct = total = 0
    for images, labels in dataloader:
        labels = labels.to(device)
        predictions = model(images.to(device)).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    if total == 0:
        raise ValueError("dataloader is empty")
    return correct / total
