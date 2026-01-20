"""Training loop for acoustic models.

This module provides the training function for acoustic models (Ua_w2v).
It mirrors the structure of swp/train/repetition.py but handles the
acoustic-specific data format (features, targets, lengths).
"""

import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..models.autoencoder import Unimodel
from ..utils.datasets import get_phoneme_to_id
from ..utils.grid_search import grid_search_log
from ..utils.models import save_weights


def train(
    model: Unimodel,
    model_name: str,
    train_name: str,
    criterion: nn.Module,
    optimizer: Optimizer,
    phoneme_to_id: dict[str, int],
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int,
    device: str | torch.device,
    verbose: bool = False,
) -> None:
    r"""Trains the acoustic `model` over `num_epoch` epochs.

    This function mirrors swp/train/repetition.py but handles the acoustic
    data format: (features, targets, lengths) instead of (data, target).

    Args:
        model: Acoustic Unimodel (must have is_acoustic=True)
        model_name: Codified model name for saving
        train_name: Codified training name for saving
        criterion: Loss function
        optimizer: Optimizer
        phoneme_to_id: Phoneme vocabulary mapping
        train_loader: DataLoader yielding (features, targets, lengths)
        valid_loader: DataLoader yielding (features, targets, lengths)
        num_epochs: Number of training epochs
        device: Device to train on
        verbose: Print intermediate logs

    Checkpointing happens 10 times during the first epoch, then once after each epoch.
    """
    if not model.is_acoustic:
        raise ValueError("Acoustic model required (is_acoustic must be True)")

    model.to(device)
    model.train()

    train_losses = []
    valid_losses = []
    train_errors = []
    valid_errors = []
    epoch_times = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        if verbose:
            print(f"\nEpoch {epoch}")

        ### TRAINING LOOP ###
        model.train()
        train_loss = 0
        train_error = 0
        checkpoint = 1

        for i, (features, target, lengths) in enumerate(train_loader, 1):
            if verbose:
                print(f"{i}/{len(train_loader)}", end="\r")

            features = features.to(device)
            target = target.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()

            # Forward pass (acoustic model needs lengths)
            output = model(features, target, lengths)

            # Loss computation
            loss = criterion(output, target)
            train_loss += loss.item()

            # Error computation
            preds = torch.argmax(output[0], dim=-1)
            mask = target != phoneme_to_id["<PAD>"]
            train_error += torch.any((preds != target) * mask, dim=1).sum().item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Checkpointing during first epoch
            if epoch == 1 and checkpoint != 10 and i % ((len(train_loader) // 10) or 1) == 0:
                save_weights(model_name, train_name, model, epoch, checkpoint)
                if verbose:
                    print(f"Checkpoint {checkpoint}: {(train_loss / i):.3f}")
                checkpoint += 1

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_errors.append(train_error)
        if verbose:
            if train_loss >= 0.001:
                print(f"Train Loss: {train_loss:.3f}")
            else:
                print(f"Train Loss: {train_loss:.2e}")

        ### VALIDATION LOOP ###
        model.eval()
        valid_loss = 0
        valid_error = 0

        with torch.no_grad():
            for i, (features, target, lengths) in enumerate(valid_loader, 1):
                if verbose:
                    print(f"{i+1}/{len(valid_loader)}", end="\r")

                features = features.to(device)
                target = target.to(device)
                lengths = lengths.to(device)

                # Forward pass
                output = model(features, target, lengths)

                # Loss computation
                loss = criterion(output, target)
                valid_loss += loss.item()

                # Error computation
                preds = torch.argmax(output[0], dim=-1)
                mask = target != phoneme_to_id["<PAD>"]
                valid_error += torch.any((preds != target) * mask, dim=1).sum().item()

        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        valid_errors.append(valid_error)
        if verbose:
            if valid_loss >= 0.001:
                print(f"Valid Loss: {valid_loss:.3f}")
            else:
                print(f"Valid Loss: {valid_loss:.2e}")

        ### POST TRAIN/VALID ###
        save_weights(model_name, train_name, model=model, epoch=epoch)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        if verbose:
            print(f"Train Errors: {train_error}")
            print(f"Valid Errors: {valid_error}")
            h = epoch_time // 3600
            m = epoch_time % 3600 // 60
            s = epoch_time % 3600 % 60
            print(f"Epoch Time: {h:.0f}h {m:.0f}m {s:.0f}s")

    grid_search_log(
        train_losses,
        valid_losses,
        train_errors,
        valid_errors,
        model_name,
        train_name,
        num_epochs,
    )
