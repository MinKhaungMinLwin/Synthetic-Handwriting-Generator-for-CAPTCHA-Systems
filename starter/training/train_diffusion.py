"""
train_diffusion.py
------------------
Defines training loop and sampling utilities for the Conditional Diffusion Model.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from ..utils.checkpoint import save_checkpoint
except ImportError:  # Support notebooks launched from the starter directory.
    from utils.checkpoint import save_checkpoint


def linear_beta_schedule(timesteps):
    if timesteps < 1:
        raise ValueError("timesteps must be at least 1")
    beta_start, beta_end = 1e-4, 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


@torch.no_grad()
def sample_images(
    model,
    device,
    num_samples=16,
    num_classes=10,
    img_size=(1, 28, 28),
    timesteps=200,
    class_labels=None,
):
    """Generate class-conditioned samples with the DDPM reverse process."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)

    imgs = torch.randn(num_samples, *img_size).to(device)

    if class_labels is None:
        labels = torch.tensor(
            [i % num_classes for i in range(num_samples)], device=device
        )
    else:
        labels = torch.as_tensor(class_labels, dtype=torch.long, device=device)
        if labels.shape != (num_samples,):
            raise ValueError(
                f"class_labels must have shape ({num_samples},), got {tuple(labels.shape)}"
            )

    was_training = model.training
    model.eval()
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(imgs, t_tensor, labels)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        noise = torch.randn_like(imgs) if t > 0 else torch.zeros_like(imgs)
        imgs = (1 / torch.sqrt(alpha)) * (
            imgs - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
        ) + torch.sqrt(betas[t]) * noise

    if was_training:
        model.train()
    return imgs.clamp(-1, 1), labels


def train_diffusion(
    model, dataloader, device, num_classes, timesteps=200, epochs=20, lr=1e-4,
    checkpoint_dir="../checkpoints", checkpoint_every=5,
):
    """Train a conditional DDPM denoiser using the noise-prediction objective."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if len(dataloader) < 1:
        raise ValueError("dataloader must contain at least one batch")
    del num_classes  # Labels come from the dataset; retained for API compatibility.
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)

    history = []
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()
            noise = torch.randn_like(imgs)
            sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t])[
                :, None, None, None
            ]
            noisy_imgs = sqrt_alpha_bar * imgs + sqrt_one_minus_alpha_bar * noise

            pred_noise = model(noisy_imgs, t, labels)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader)
        history.append(epoch_loss)
        print(f"Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}")

        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, epoch_loss, name="diffusion_unet",
                path=checkpoint_dir,
            )

    save_checkpoint(
        model, optimizer, epoch="final", loss=history[-1],
        name="diffusion_unet", path=checkpoint_dir,
    )
    return history
