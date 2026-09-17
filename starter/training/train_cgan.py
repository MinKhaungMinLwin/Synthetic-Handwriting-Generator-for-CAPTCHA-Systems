"""
train_cgan.py
-------------
Reusable training function for cGAN experiments.
"""

import torch

try:
    from ..utils.checkpoint import save_checkpoint
except ImportError:  # Support notebooks launched from the starter directory.
    from utils.checkpoint import save_checkpoint


def train_cgan(
    generator,
    discriminator,
    dataloader,
    optimizer_G,
    optimizer_D,
    criterion,
    device,
    z_dim,
    num_classes,
    epochs=50,
    checkpoint_dir="../checkpoints",
    checkpoint_every=10,
):
    """Train a conditional GAN and return per-epoch generator/discriminator losses."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if len(dataloader) < 1:
        raise ValueError("dataloader must contain at least one batch")
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()
    history = {"generator_loss": [], "discriminator_loss": []}

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, z_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)
            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs, labels), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)
            # Real images paired with an incorrect class are also fake pairs.
            # This prevents D from ignoring the conditioning label and gives G
            # a much stronger class-specific training signal.
            label_offsets = torch.randint(1, num_classes, (batch_size,), device=device)
            wrong_labels = (labels + label_offsets) % num_classes
            mismatch_loss = criterion(discriminator(imgs, wrong_labels), fake)
            d_loss = (real_loss + fake_loss + mismatch_loss) / 3
            d_loss.backward()
            optimizer_D.step()

        print(
            f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}"
        )
        history["generator_loss"].append(g_loss.item())
        history["discriminator_loss"].append(d_loss.item())

        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                generator, optimizer_G, epoch + 1, g_loss.item(),
                name="cgan_generator", path=checkpoint_dir,
            )
            save_checkpoint(
                discriminator, optimizer_D, epoch + 1, d_loss.item(),
                name="cgan_discriminator", path=checkpoint_dir,
            )
            print(f"Saved checkpoints at epoch {epoch+1}")

    save_checkpoint(
        generator, optimizer_G, "final", history["generator_loss"][-1],
        name="cgan_generator", path=checkpoint_dir,
    )
    save_checkpoint(
        discriminator, optimizer_D, "final", history["discriminator_loss"][-1],
        name="cgan_discriminator", path=checkpoint_dir,
    )
    return history
