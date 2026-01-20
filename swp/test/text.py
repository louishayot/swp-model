from typing import Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models.autoencoder import Bimodel, Unimodel
from ..models.metrics import classic_errors
from ..utils.datasets import get_phoneme_to_id


def test(
    model: Unimodel | Bimodel,
    device: str | torch.device,
    test_df: pd.DataFrame,
    test_loader: DataLoader,
    include_stress: bool = False,
    error_meter: Callable[[torch.Tensor, torch.Tensor], int] = classic_errors,
    verbose: bool = False,
) -> tuple[pd.DataFrame, float]:
    r"""Takes any pd.df with Word and Phonemes columns, returns same df with phoneme predictions.

    For text models (Ut): input is characters, output is phonemes.
    """

    if isinstance(model, Bimodel):
        raise ValueError("Text testing requires Unimodel, got Bimodel")

    if isinstance(model, Unimodel) and not model.is_text:
        raise ValueError("Text testing requires model.is_text == True")

    test_error = 0
    last_index = 0
    predictions = []
    phoneme_key = "Phonemes" if include_stress else "No_Stress"
    phoneme_to_id = get_phoneme_to_id(include_stress)
    id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(test_loader, 1):

            if verbose:
                print(f"{i+1}/{len(test_loader)+1}   ", end="\r")

            inputs = inputs.to(device)
            target = target.to(device).long()

            # Forward pass
            output = model(inputs, target)
            preds = torch.argmax(output[0], dim=-1)

            # Error computation
            test_error += error_meter(preds, target)

            # Save predictions
            batch_size = target.shape[0]
            for j in range(batch_size):
                ground_truth = test_df.iloc[last_index + j][phoneme_key]
                phonemes = [id_to_phoneme[int(idx)] for idx in preds[j, : len(ground_truth)]]
                predictions.append(phonemes)
            last_index += batch_size

    test_df["Prediction"] = predictions

    if verbose:
        print(f"test error: {test_error}/{len(test_df)}")

    return test_df, test_error
