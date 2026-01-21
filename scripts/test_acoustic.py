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


def find_latest_checkpoint(weights_dir: Path, model_name: str, train_name: str) -> str:
    """Find the latest checkpoint number in a training directory."""
    train_dir = weights_dir / model_name / train_name
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # Find all .pth files that are just epoch numbers (not checkpoints like 1_2.pth)
    checkpoints = []
    for f in train_dir.glob("*.pth"):
        name = f.stem
        if name.isdigit():
            checkpoints.append(int(name))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {train_dir}")

    return str(max(checkpoints))


def normalize_manifest_path(path_str: str) -> str:
    """Normalize a manifest path for comparison."""
    if not path_str:
        return ""
    # Resolve to absolute, then get just the key parts
    p = Path(path_str).resolve()
    return str(p)


def load_run_info(
    weights_dir: Path,
    model_name: str = None,
    train_name: str = None,
    manifest_path: str = None,
):
    """Load run info from a training directory, or search for it.

    If manifest_path is provided, only consider runs whose manifest_path matches.
    If multiple runs match, pick the most recently modified.
    """
    import json

    if model_name and train_name:
        run_info_path = weights_dir / model_name / train_name / "run_info.json"
        if run_info_path.exists():
            with open(run_info_path) as f:
                return json.load(f)

    # Search for acoustic models if not specified
    if model_name is None:
        candidates = []
        target_manifest = normalize_manifest_path(manifest_path) if manifest_path else None

        for model_dir in weights_dir.glob("Ua_w2v_*"):
            if model_dir.is_dir():
                for train_dir in model_dir.iterdir():
                    if train_dir.is_dir():
                        run_info_path = train_dir / "run_info.json"
                        if run_info_path.exists():
                            with open(run_info_path) as f:
                                info = json.load(f)

                            # If manifest_path filter is set, check for match
                            if target_manifest:
                                run_manifest = normalize_manifest_path(info.get("manifest_path", ""))
                                if run_manifest != target_manifest:
                                    continue  # Skip non-matching runs

                            # Get mtime for sorting
                            mtime = run_info_path.stat().st_mtime
                            candidates.append((mtime, info, run_info_path))

        if candidates:
            # Sort by mtime descending (most recent first)
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_info, best_path = candidates[0]
            print(f"Auto-detected run: {best_info['model_name']}~{best_info['train_name']}")
            if len(candidates) > 1:
                print(f"  (selected from {len(candidates)} matching runs by most recent mtime)")
            return best_info

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test acoustic encoder model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name string (auto-detected from run_info.json if not specified)",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        default=None,
        help="Training name string (auto-detected from run_info.json if not specified)",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Path to test manifest JSON file (uses training manifest if not specified)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load (e.g., '10' for epoch 10). Uses latest if not specified.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to training run directory (alternative to model_name + train_name)",
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
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print behavioral signature statistics (length distribution, EOS position)",
    )

    args = parser.parse_args()

    # Setup
    seed_everything(args.seed)
    backend_setup()
    device = set_device()

    # Import paths helper
    from swp.utils.paths import get_weights_dir
    weights_dir = get_weights_dir()

    # Resolve model_name and train_name
    model_name = args.model_name
    train_name = args.train_name
    manifest_path = args.manifest_path
    include_stress = args.include_stress

    # Try to load from run_dir if specified
    if args.run_dir:
        import json
        run_dir = Path(args.run_dir)
        run_info_path = run_dir / "run_info.json"
        if run_info_path.exists():
            with open(run_info_path) as f:
                run_info = json.load(f)
            model_name = model_name or run_info["model_name"]
            train_name = train_name or run_info["train_name"]
            manifest_path = manifest_path or run_info.get("manifest_path")
            include_stress = run_info.get("include_stress", include_stress)

    # If still missing, try to auto-detect from run_info.json
    # Pass manifest_path to filter runs that match the provided manifest
    if not model_name or not train_name:
        run_info = load_run_info(weights_dir, model_name, train_name, manifest_path)
        if run_info:
            model_name = model_name or run_info["model_name"]
            train_name = train_name or run_info["train_name"]
            manifest_path = manifest_path or run_info.get("manifest_path")
            include_stress = run_info.get("include_stress", include_stress)
        else:
            if manifest_path:
                parser.error(f"No runs found matching manifest: {manifest_path}\nPlease specify --model_name and --train_name")
            else:
                parser.error("Could not auto-detect model. Please specify --model_name and --train_name")

    # Auto-detect checkpoint if not specified
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = find_latest_checkpoint(weights_dir, model_name, train_name)
        print(f"Auto-detected latest checkpoint: {checkpoint}")

    # Require manifest_path
    if not manifest_path:
        parser.error("--manifest_path is required (or use --run_dir to load from run_info.json)")

    # Load model
    print(f"Loading model: {model_name}")
    print(f"Training: {train_name}")
    print(f"Checkpoint: {checkpoint}")

    model = get_model(model_name)
    load_weights(model, model_name, train_name, checkpoint, device)

    # Load test data
    phoneme_to_id = get_phoneme_to_id(include_stress)
    test_loader = get_acoustic_testloader(
        manifest_path=Path(manifest_path),
        batch_size=args.batch_size,
        include_stress=include_stress,
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
    print(f"Results for {model_name}")
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

    # Behavioral signature statistics
    if args.stats and results:
        print("\n" + "=" * 60)
        print("Behavioral Signature Statistics")
        print("=" * 60)

        # Compute length statistics
        target_lengths = [len(r["target"]) for r in results]
        pred_lengths = [len(r["prediction"]) for r in results]

        # Find EOS positions (1-indexed, or None if not found)
        def find_eos_position(seq):
            try:
                return seq.index("<EOS>") + 1
            except ValueError:
                return None

        target_eos_positions = [find_eos_position(r["target"]) for r in results]
        pred_eos_positions = [find_eos_position(r["prediction"]) for r in results]

        # Filter out None values for statistics
        valid_target_eos = [p for p in target_eos_positions if p is not None]
        valid_pred_eos = [p for p in pred_eos_positions if p is not None]

        print("\nLength Distribution:")
        print(f"  Target lengths:     min={min(target_lengths)}, max={max(target_lengths)}, mean={sum(target_lengths)/len(target_lengths):.2f}")
        print(f"  Prediction lengths: min={min(pred_lengths)}, max={max(pred_lengths)}, mean={sum(pred_lengths)/len(pred_lengths):.2f}")

        # Length match rate
        length_matches = sum(1 for t, p in zip(target_lengths, pred_lengths) if t == p)
        print(f"  Length match rate:  {length_matches}/{len(results)} ({100*length_matches/len(results):.1f}%)")

        print("\nEOS Position Statistics:")
        if valid_target_eos:
            print(f"  Target EOS position:  min={min(valid_target_eos)}, max={max(valid_target_eos)}, mean={sum(valid_target_eos)/len(valid_target_eos):.2f}")
        if valid_pred_eos:
            print(f"  Pred EOS position:    min={min(valid_pred_eos)}, max={max(valid_pred_eos)}, mean={sum(valid_pred_eos)/len(valid_pred_eos):.2f}")
            print(f"  Samples with EOS:     {len(valid_pred_eos)}/{len(results)} ({100*len(valid_pred_eos)/len(results):.1f}%)")

        # EOS position match rate
        eos_matches = sum(1 for t, p in zip(target_eos_positions, pred_eos_positions) if t == p and t is not None)
        print(f"  EOS position match:   {eos_matches}/{len(results)} ({100*eos_matches/len(results):.1f}%)")

        # Length distribution histogram (buckets)
        print("\nPrediction Length Histogram:")
        from collections import Counter
        length_counts = Counter(pred_lengths)
        for length in sorted(length_counts.keys()):
            bar = "#" * min(length_counts[length], 50)
            print(f"  {length:3d}: {bar} ({length_counts[length]})")
