"""
checkpoint.py
--------------
Utility functions for saving and loading model checkpoints consistently.
"""

from pathlib import Path
import torch

def save_checkpoint(model, optimizer=None, epoch=None, loss=None, name="model", path="../checkpoints"):
    """Save model and optional optimizer state, returning the checkpoint path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    epoch_name = "final" if epoch is None else str(epoch)
    save_path = directory / f"{name}_epoch{epoch_name}.pt"

    state = {"model_state": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if loss is not None:
        state["loss"] = loss

    torch.save(state, save_path)
    print(f"Saved checkpoint: {save_path}")
    return str(save_path)

def load_checkpoint(model, optimizer=None, path=None, map_location="cpu"):
    """Load either a full project checkpoint or a plain PyTorch state dict."""
    if path is None:
        raise ValueError("path is required")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded model weights from {path}")
    return model
