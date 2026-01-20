#!/usr/bin/env python3
"""Testing script for acoustic models (Ua_w2v).

This script evaluates trained acoustic encoder models on test data.

Usage:
    python scripts/test_acoustic.py \
        --model_name Ua_w2v_LSTM_h128_l1_v43_d0.0_t0.0_s1__gi768 \
        --train_name b32_l0.001_fall_s42_sn_ec \
        --manifest_path ./acoustic_features_test/manifest.json \
        --checkpoint 10 \
        --verbose
"""

import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
from pathlib import Path

import torch

from swp.datasets.acoustic import get_acoustic_testloader
from swp.utils.datasets import get_phoneme_to_id
from swp.utils.models import get_model, load_weights
from swp.utils.setup import backend_setup, seed_everything, set_device


def test_acoustic(
    model,
    test_loader,
    phoneme_to_id: dict[str, int],
    device: torch.device,
    verbose: bool = False,
) -> tuple[float, int, int]:
    """Test an acoustic model on the given dataloader.

    Args:
        model: Acoustic Unimodel
        test_loader: DataLoader yielding (features, targets, lengths, utterance_ids)
        phoneme_to_id: Phoneme vocabulary
        device: Device to run on
        verbose: Print progress

    Returns:
        tuple of (avg_loss, total_errors, total_samples)
    """
    model.eval()
    model.to(device)

    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    pad_idx = phoneme_to_id["<PAD>"]
    eos_idx = phoneme_to_id["<EOS>"]

    total_loss = 0
    total_errors = 0
    total_samples = 0

    results = []

    with torch.no_grad():
        for i, (features, target, lengths, utterance_ids) in enumerate(test_loader, 1):
            if verbose:
                print(f"{i}/{len(test_loader)}", end="\r")

            features = features.to(device)
            target = target.to(device)
            lengths = lengths.to(device)

            # Forward pass
            output = model(features, target, lengths)

            # Get predictions
            preds = torch.argmax(output[0], dim=-1)

            # Compute errors
            mask = target != pad_idx
            batch_errors = torch.any((preds != target) * mask, dim=1).sum().item()
            total_errors += batch_errors
            total_samples += target.size(0)

            # Decode predictions for analysis
            for j in range(target.size(0)):
                pred_tokens = preds[j].cpu().tolist()
                target_tokens = target[j].cpu().tolist()

                # Remove padding and convert to phonemes
                pred_phonemes = []
                for tok in pred_tokens:
                    if tok == eos_idx:
                        pred_phonemes.append("<EOS>")
                        break
                    elif tok != pad_idx:
                        pred_phonemes.append(id_to_phoneme.get(tok, "?"))

                target_phonemes = []
                for tok in target_tokens:
                    if tok == eos_idx:
                        target_phonemes.append("<EOS>")
                        break
                    elif tok != pad_idx:
                        target_phonemes.append(id_to_phoneme.get(tok, "?"))

                results.append({
                    "utterance_id": utterance_ids[j],
                    "target": target_phonemes,
                    "prediction": pred_phonemes,
                    "correct": pred_phonemes == target_phonemes,
                })

    accuracy = (total_samples - total_errors) / total_samples if total_samples > 0 else 0

    return accuracy, total_errors, total_samples, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test acoustic encoder model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name string",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        required=True,
        help="Training name string",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to test manifest JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint to load (e.g., '10' for epoch 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="Maximum frames per sample",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and examples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup
    seed_everything(args.seed)
    backend_setup()
    device = set_device()

    # Load model
    print(f"Loading model: {args.model_name}")
    print(f"Training: {args.train_name}")
    print(f"Checkpoint: {args.checkpoint}")

    model = get_model(args.model_name)
    load_weights(model, args.model_name, args.train_name, args.checkpoint, device)

    # Load test data
    phoneme_to_id = get_phoneme_to_id(args.include_stress)
    test_loader = get_acoustic_testloader(
        manifest_path=Path(args.manifest_path),
        batch_size=args.batch_size,
        include_stress=args.include_stress,
        max_frames=args.max_frames,
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    # Run test
    accuracy, total_errors, total_samples, results = test_acoustic(
        model=model,
        test_loader=test_loader,
        phoneme_to_id=phoneme_to_id,
        device=device,
        verbose=args.verbose,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"Results for {args.model_name}")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Total errors: {total_errors}")
    print(f"Accuracy: {accuracy:.4f} ({(1-accuracy)*100:.2f}% error rate)")

    if args.verbose and results:
        print("\nSample predictions:")
        print("-" * 40)
        for i, r in enumerate(results[:5]):
            print(f"  [{r['utterance_id']}]")
            print(f"    Target:     {' '.join(r['target'])}")
            print(f"    Prediction: {' '.join(r['prediction'])}")
            print(f"    Correct:    {r['correct']}")
            print()
