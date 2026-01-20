"""Acoustic dataset for training with wav2vec2/HuBERT features.

This module provides dataset classes and dataloaders for training
acoustic models on pre-extracted wav2vec2/HuBERT features paired
with phoneme transcriptions.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nested import nested_tensor
from torch.utils.data import DataLoader, Dataset

from ..utils.datasets import get_phoneme_to_id


class AcousticTrainDataset(Dataset):
    """Dataset for acoustic features with phoneme targets.

    Loads pre-extracted acoustic features (wav2vec2/HuBERT) from disk
    and pairs them with phoneme transcriptions for training.

    Args:
        manifest_path: Path to manifest JSON file containing utterance info
        phoneme_to_id: Dictionary mapping phonemes to indices
        max_frames: Maximum number of frames (for memory efficiency). None = no limit.
        include_stress: Whether phonemes include stress markers

    The manifest JSON should have the structure:
    {
        "utterance_id": {
            "feature_path": "path/to/features.npz",
            "phonemes": ["P", "AH", "T", ...],
            "num_frames": 150
        },
        ...
    }
    """

    def __init__(
        self,
        manifest_path: Path | str,
        phoneme_to_id: dict[str, int],
        max_frames: int | None = None,
        include_stress: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.phoneme_to_id = phoneme_to_id
        self.max_frames = max_frames
        self.include_stress = include_stress

        # Load manifest
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)

        self.utterance_ids = list(self.manifest.keys())

        # Store base directory for resolving relative paths
        self.base_dir = self.manifest_path.parent

    def __len__(self) -> int:
        return len(self.utterance_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get a single sample.

        Returns:
            tuple of (features, tokens, length) where:
                - features: (time, input_dim) acoustic features
                - tokens: (seq_len,) phoneme token indices with <EOS>
                - length: actual number of frames (before any truncation)
        """
        utterance_id = self.utterance_ids[idx]
        item = self.manifest[utterance_id]

        # Load features
        feature_path = item["feature_path"]
        if not Path(feature_path).is_absolute():
            feature_path = self.base_dir / feature_path

        data = np.load(feature_path)
        features = torch.from_numpy(data["features"]).float()

        # Record original length
        original_length = features.size(0)

        # Truncate if needed
        if self.max_frames is not None and features.size(0) > self.max_frames:
            features = features[: self.max_frames]

        actual_length = features.size(0)

        # Convert phonemes to tokens
        phonemes = item["phonemes"].copy() if isinstance(item["phonemes"], list) else list(item["phonemes"])
        phonemes.append("<EOS>")

        # Map phonemes to indices, handling unknown phonemes
        tokens = []
        for p in phonemes:
            if p in self.phoneme_to_id:
                tokens.append(self.phoneme_to_id[p])
            else:
                # Try without stress marker if include_stress is False
                p_no_stress = p.rstrip("012") if not self.include_stress else p
                if p_no_stress in self.phoneme_to_id:
                    tokens.append(self.phoneme_to_id[p_no_stress])
                # Skip unknown phonemes (shouldn't happen with proper G2P)

        tokens = torch.tensor(tokens, dtype=torch.long)

        return features, tokens, actual_length


class AcousticTestDataset(Dataset):
    """Dataset for testing acoustic models.

    Similar to AcousticTrainDataset but designed for evaluation,
    optionally including additional metadata.

    Args:
        manifest_path: Path to manifest JSON file
        phoneme_to_id: Dictionary mapping phonemes to indices
        max_frames: Maximum number of frames
        include_stress: Whether phonemes include stress markers
    """

    def __init__(
        self,
        manifest_path: Path | str,
        phoneme_to_id: dict[str, int],
        max_frames: int | None = None,
        include_stress: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.phoneme_to_id = phoneme_to_id
        self.max_frames = max_frames
        self.include_stress = include_stress

        with open(self.manifest_path) as f:
            self.manifest = json.load(f)

        self.utterance_ids = list(self.manifest.keys())
        self.base_dir = self.manifest_path.parent

    def __len__(self) -> int:
        return len(self.utterance_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        """Get a single sample with utterance ID for evaluation.

        Returns:
            tuple of (features, tokens, length, utterance_id)
        """
        utterance_id = self.utterance_ids[idx]
        item = self.manifest[utterance_id]

        # Load features
        feature_path = item["feature_path"]
        if not Path(feature_path).is_absolute():
            feature_path = self.base_dir / feature_path

        data = np.load(feature_path)
        features = torch.from_numpy(data["features"]).float()

        # Truncate if needed
        if self.max_frames is not None and features.size(0) > self.max_frames:
            features = features[: self.max_frames]

        actual_length = features.size(0)

        # Convert phonemes to tokens
        phonemes = item["phonemes"].copy() if isinstance(item["phonemes"], list) else list(item["phonemes"])
        phonemes.append("<EOS>")

        tokens = []
        for p in phonemes:
            if p in self.phoneme_to_id:
                tokens.append(self.phoneme_to_id[p])
            else:
                p_no_stress = p.rstrip("012") if not self.include_stress else p
                if p_no_stress in self.phoneme_to_id:
                    tokens.append(self.phoneme_to_id[p_no_stress])

        tokens = torch.tensor(tokens, dtype=torch.long)

        return features, tokens, actual_length, utterance_id


def acoustic_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
    pad_value: int,
    feature_pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for variable-length acoustic features and phoneme targets.

    Args:
        batch: List of (features, tokens, length) tuples
        pad_value: Padding value for phoneme tokens (usually <PAD> index)
        feature_pad_value: Padding value for acoustic features

    Returns:
        tuple of (padded_features, padded_targets, lengths) where:
            - padded_features: (batch, max_time, feature_dim)
            - padded_targets: (batch, max_seq_len)
            - lengths: (batch,) actual frame lengths
    """
    features, targets, lengths = zip(*batch)

    # Pad features (time dimension)
    max_time = max(f.size(0) for f in features)
    feature_dim = features[0].size(1)
    padded_features = torch.full(
        (len(features), max_time, feature_dim),
        feature_pad_value,
        dtype=torch.float,
    )
    for i, f in enumerate(features):
        padded_features[i, : f.size(0)] = f

    # Pad targets using nested tensor (consistent with phoneme collate)
    nt_targets = nested_tensor(list(targets), dtype=torch.long)
    padded_targets = nt_targets.to_padded_tensor(padding=pad_value)

    # Convert lengths to tensor
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return padded_features, padded_targets, lengths_tensor


def acoustic_test_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, str]],
    pad_value: int,
    feature_pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Collate function for test dataset with utterance IDs.

    Args:
        batch: List of (features, tokens, length, utterance_id) tuples
        pad_value: Padding value for phoneme tokens
        feature_pad_value: Padding value for acoustic features

    Returns:
        tuple of (padded_features, padded_targets, lengths, utterance_ids)
    """
    features, targets, lengths, utterance_ids = zip(*batch)

    max_time = max(f.size(0) for f in features)
    feature_dim = features[0].size(1)
    padded_features = torch.full(
        (len(features), max_time, feature_dim),
        feature_pad_value,
        dtype=torch.float,
    )
    for i, f in enumerate(features):
        padded_features[i, : f.size(0)] = f

    nt_targets = nested_tensor(list(targets), dtype=torch.long)
    padded_targets = nt_targets.to_padded_tensor(padding=pad_value)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return padded_features, padded_targets, lengths_tensor, list(utterance_ids)


def get_acoustic_trainloader(
    manifest_path: Path | str,
    batch_size: int,
    include_stress: bool = False,
    max_frames: int | None = 500,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Return a dataloader for acoustic training data.

    Args:
        manifest_path: Path to manifest JSON file
        batch_size: Batch size
        include_stress: Whether phonemes include stress markers
        max_frames: Maximum number of frames per sample
        shuffle: Whether to shuffle the data
        generator: Random generator for shuffling

    Returns:
        DataLoader yielding (features, targets, lengths) batches
    """
    phoneme_to_id = get_phoneme_to_id(include_stress)
    dataset = AcousticTrainDataset(
        manifest_path=manifest_path,
        phoneme_to_id=phoneme_to_id,
        max_frames=max_frames,
        include_stress=include_stress,
    )

    if generator is None:
        generator = torch.Generator().manual_seed(42)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: acoustic_collate_fn(
            batch, pad_value=phoneme_to_id["<PAD>"]
        ),
    )


def get_acoustic_testloader(
    manifest_path: Path | str,
    batch_size: int,
    include_stress: bool = False,
    max_frames: int | None = 500,
) -> DataLoader:
    """Return a dataloader for acoustic test data.

    Args:
        manifest_path: Path to manifest JSON file
        batch_size: Batch size
        include_stress: Whether phonemes include stress markers
        max_frames: Maximum number of frames per sample

    Returns:
        DataLoader yielding (features, targets, lengths, utterance_ids) batches
    """
    phoneme_to_id = get_phoneme_to_id(include_stress)
    dataset = AcousticTestDataset(
        manifest_path=manifest_path,
        phoneme_to_id=phoneme_to_id,
        max_frames=max_frames,
        include_stress=include_stress,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: acoustic_test_collate_fn(
            batch, pad_value=phoneme_to_id["<PAD>"]
        ),
    )
