"""Dataset classes for text (character) to phoneme mapping."""
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nested import nested_tensor
from torch.utils.data import DataLoader, Dataset

from ..utils.datasets import (
    get_char_to_id,
    get_epoch_numpy,
    get_evaluation_dataset,
    get_phoneme_to_id,
    get_train_fold,
    get_valid_fold,
)


class TextTrainDataset(Dataset):
    """Dataset for text-to-phoneme training.

    Input: character IDs of word (lowercase) with <UNK> fallback + <EOS>
    Target: phoneme IDs + <EOS>

    Args:
        fold_id: fold number to load, if None returns complete training set
        train: return training split if True, validation split otherwise
        char_to_id: dict mapping characters to int for tokenization
        phoneme_to_id: dict mapping phonemes to int for tokenization
        include_stress: if True, phonemes will include stress
    """

    def __init__(
        self,
        fold_id: int | None,
        train: bool,
        char_to_id: dict[str, int],
        phoneme_to_id: dict[str, int],
        include_stress: bool = False,
    ):
        self.fold_id = fold_id
        self.train = train
        if self.train:
            self.data_df = get_train_fold(fold_id)
            self.epoch_ids = get_epoch_numpy(fold_id=fold_id, epoch_size=int(1e6))
        else:
            self.data_df = get_valid_fold(self.fold_id)
            self.epoch_ids = np.arange(len(self.data_df))
        self.char_to_id = char_to_id
        self.phoneme_to_id = phoneme_to_id
        self.phoneme_key = "Phonemes" if include_stress else "No_Stress"

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.data_df.iloc[self.epoch_ids[index]]

        # Input: characters (with <UNK> fallback)
        word: str = row["Word"].lower()
        unk_id = self.char_to_id["<UNK>"]
        chars = [self.char_to_id.get(c, unk_id) for c in word]
        chars.append(self.char_to_id["<EOS>"])
        char_ids = torch.tensor(chars, dtype=torch.long)

        # Target: phonemes
        phonemes: list[str] = row[self.phoneme_key].copy()
        phonemes.append("<EOS>")
        phoneme_ids = torch.tensor(
            [self.phoneme_to_id[p] for p in phonemes], dtype=torch.long
        )

        return char_ids, phoneme_ids

    def __len__(self) -> int:
        return len(self.epoch_ids)


def text_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    char_pad: int,
    phoneme_pad: int,
):
    """Collate with separate pad values for input (chars) and target (phonemes)."""
    data, target = tuple(zip(*batch))
    padded_data = nested_tensor(list(data), dtype=torch.long).to_padded_tensor(
        padding=char_pad
    )
    padded_target = nested_tensor(list(target), dtype=torch.long).to_padded_tensor(
        padding=phoneme_pad
    )
    return padded_data, padded_target


def get_text_trainloader(
    fold_id: int | None,
    train: bool,
    batch_size: int,
    generator: torch.Generator | None = None,
    include_stress: bool = False,
) -> DataLoader:
    """Return DataLoader for text-to-phoneme training.

    If `include_stress` is True, phonemes will include stress.
    Return training data if `train` is True, validation data otherwise.
    If `fold_id` is None, returns the complete training set.
    """
    char_to_id = get_char_to_id()
    phoneme_to_id = get_phoneme_to_id(include_stress)

    dataset = TextTrainDataset(
        fold_id=fold_id,
        train=train,
        char_to_id=char_to_id,
        phoneme_to_id=phoneme_to_id,
        include_stress=include_stress,
    )

    if generator is None:
        generator = torch.Generator().manual_seed(42)

    return DataLoader(
        dataset,
        batch_size,
        shuffle=train,
        generator=generator,
        collate_fn=lambda b: text_collate_fn(
            b, char_pad=char_to_id["<PAD>"], phoneme_pad=phoneme_to_id["<PAD>"]
        ),
    )


class TextTestDataset(Dataset):
    """Dataset for text-to-phoneme testing.

    Args:
        char_to_id: dict mapping characters to int for tokenization
        phoneme_to_id: dict mapping phonemes to int for tokenization
        include_stress: if True, phonemes will include stress
        dataset_df: optional DataFrame to override test data
    """

    def __init__(
        self,
        char_to_id: dict[str, int],
        phoneme_to_id: dict[str, int],
        include_stress: bool = False,
        dataset_df: pd.DataFrame | None = None,
    ):
        if dataset_df is None:
            self.data_df = get_evaluation_dataset()
        else:
            self.data_df = dataset_df
        self.epoch_ids = np.arange(len(self.data_df))
        self.char_to_id = char_to_id
        self.phoneme_to_id = phoneme_to_id
        self.phoneme_key = "Phonemes" if include_stress else "No_Stress"

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.data_df.iloc[self.epoch_ids[index]]

        # Input: characters (with <UNK> fallback)
        word: str = row["Word"].lower()
        unk_id = self.char_to_id["<UNK>"]
        chars = [self.char_to_id.get(c, unk_id) for c in word]
        chars.append(self.char_to_id["<EOS>"])
        char_ids = torch.tensor(chars, dtype=torch.long)

        # Target: phonemes
        phonemes: list[str] = row[self.phoneme_key].copy()
        phonemes.append("<EOS>")
        phoneme_ids = torch.tensor(
            [self.phoneme_to_id[p] for p in phonemes], dtype=torch.long
        )

        return char_ids, phoneme_ids

    def __len__(self) -> int:
        return len(self.epoch_ids)


def get_text_testloader(
    batch_size: int,
    include_stress: bool = False,
    dataset_df: pd.DataFrame | None = None,
) -> DataLoader:
    """Return DataLoader for text-to-phoneme testing.

    If `include_stress` is True, phonemes will include stress.
    Pass a DataFrame as `dataset_df` to override the test data.
    """
    char_to_id = get_char_to_id()
    phoneme_to_id = get_phoneme_to_id(include_stress)

    dataset = TextTestDataset(
        char_to_id=char_to_id,
        phoneme_to_id=phoneme_to_id,
        include_stress=include_stress,
        dataset_df=dataset_df,
    )

    return DataLoader(
        dataset,
        batch_size,
        collate_fn=lambda b: text_collate_fn(
            b, char_pad=char_to_id["<PAD>"], phoneme_pad=phoneme_to_id["<PAD>"]
        ),
    )
