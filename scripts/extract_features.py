#!/usr/bin/env python3
"""Extract wav2vec2 features from LibriSpeech via HuggingFace datasets.

This script extracts acoustic features from LibriSpeech audio using
a frozen wav2vec2 model and saves them to disk for offline training.

Usage:
    python scripts/extract_features.py \
        --output_dir ./acoustic_features \
        --split train.clean.100 \
        --limit 1000

Features are saved as .npz files with a manifest.json for the dataloader.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from swp.utils.datasets import get_phoneme_to_id


def load_wav2vec2_model(model_name: str, device: torch.device):
    """Load wav2vec2 model and processor from HuggingFace."""
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return processor, model


def load_g2p():
    """Load grapheme-to-phoneme converter."""
    from g2p_en import G2p

    return G2p()


def text_to_phonemes(text: str, g2p, phoneme_to_id: dict, include_stress: bool = False) -> list[str]:
    """Convert text to phoneme sequence using G2P.

    Args:
        text: Input text (e.g., "hello world")
        g2p: G2p instance
        phoneme_to_id: Phoneme vocabulary dictionary
        include_stress: Whether to keep stress markers

    Returns:
        List of phonemes that exist in the vocabulary
    """
    # Run G2P
    raw_phonemes = g2p(text)

    # Filter and normalize phonemes
    phonemes = []
    for p in raw_phonemes:
        # Skip spaces and punctuation
        if p.strip() == "" or p in " .,!?;:'-\"":
            continue

        # Normalize phoneme
        if not include_stress:
            # Remove stress markers (digits at end)
            p_clean = p.rstrip("012")
        else:
            p_clean = p

        # Check if phoneme is in vocabulary
        if p_clean in phoneme_to_id:
            phonemes.append(p_clean)
        elif p.upper() in phoneme_to_id:
            phonemes.append(p.upper())
        elif p_clean.upper() in phoneme_to_id:
            phonemes.append(p_clean.upper())
        # Skip unknown phonemes

    return phonemes


def extract_features_batch(
    audio_batch: list[np.ndarray],
    processor,
    model,
    device: torch.device,
    sampling_rate: int = 16000,
) -> list[np.ndarray]:
    """Extract wav2vec2 features from a batch of audio.

    Args:
        audio_batch: List of audio arrays (1D, 16kHz)
        processor: Wav2Vec2Processor
        model: Wav2Vec2Model
        device: torch device
        sampling_rate: Audio sampling rate

    Returns:
        List of feature arrays, each (time, feature_dim)
    """
    # Process audio
    inputs = processor(
        audio_batch,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    # Move to device
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Extract features
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        features = outputs.last_hidden_state  # (batch, time, hidden_dim)

    # Convert to numpy and split by actual lengths
    features_np = features.cpu().numpy()

    # If we have attention mask, use it to get actual lengths
    if attention_mask is not None:
        # Attention mask is on input, but features are downsampled
        # wav2vec2 downsamples by factor of ~320
        input_lengths = attention_mask.sum(dim=1).cpu().numpy()
        # Approximate feature lengths (wav2vec2 uses conv with stride)
        feature_lengths = (input_lengths - 1) // 320 + 1
    else:
        feature_lengths = [f.shape[0] for f in features_np]

    # Trim padding from each sample
    result = []
    for i, length in enumerate(feature_lengths):
        length = min(int(length), features_np[i].shape[0])
        result.append(features_np[i][:length])

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract wav2vec2 features from LibriSpeech"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for features and manifest",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/wav2vec2-base",
        help="wav2vec2 model name from HuggingFace",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train.clean.100",
        help="LibriSpeech split (e.g., train.clean.100, validation.clean)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress markers in phonemes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    # Set up device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"Loading wav2vec2 model: {args.model}")
    processor, model = load_wav2vec2_model(args.model, device)

    print("Loading G2P model...")
    g2p = load_g2p()

    # Load phoneme vocabulary
    phoneme_to_id = get_phoneme_to_id(args.include_stress)
    print(f"Phoneme vocabulary size: {len(phoneme_to_id)}")

    # Load LibriSpeech dataset
    print(f"Loading LibriSpeech split: {args.split}")
    from datasets import load_dataset

    # Convert split name (train.clean.100 -> train-clean-100 for newer datasets API)
    split_name = args.split.replace(".", "-")

    try:
        # Try the standard LibriSpeech dataset
        dataset = load_dataset(
            "librispeech_asr",
            split=split_name,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading split '{split_name}': {e}")
        print("Trying alternative split format...")
        # Try with different split format
        if "train" in args.split:
            dataset = load_dataset(
                "librispeech_asr",
                "clean",
                split="train.100",
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset(
                "librispeech_asr",
                "clean",
                split="validation",
                trust_remote_code=True,
            )

    # Limit samples if specified
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Processing {len(dataset)} samples")

    # Process samples
    manifest = {}
    feature_dir = output_dir / "features"
    feature_dir.mkdir(exist_ok=True)

    for idx, sample in enumerate(tqdm(dataset, desc="Extracting features")):
        # Get audio and text
        audio = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        text = sample["text"]
        utterance_id = f"{sample['id']}" if "id" in sample else f"utt_{idx:06d}"

        # Resample if needed (wav2vec2 expects 16kHz)
        if sampling_rate != 16000:
            import torchaudio.functional as F

            audio_tensor = torch.from_numpy(audio).float()
            audio = F.resample(audio_tensor, sampling_rate, 16000).numpy()

        # Extract features (single sample)
        features_list = extract_features_batch(
            [audio], processor, model, device, sampling_rate=16000
        )
        features = features_list[0]

        # Convert text to phonemes
        phonemes = text_to_phonemes(text, g2p, phoneme_to_id, args.include_stress)

        # Skip if no valid phonemes
        if len(phonemes) == 0:
            print(f"Warning: No valid phonemes for utterance {utterance_id}, skipping")
            continue

        # Save features
        feature_path = feature_dir / f"{utterance_id}.npz"
        np.savez_compressed(feature_path, features=features)

        # Add to manifest
        manifest[utterance_id] = {
            "feature_path": f"features/{utterance_id}.npz",
            "phonemes": phonemes,
            "num_frames": features.shape[0],
            "text": text,
        }

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExtraction complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Total samples: {len(manifest)}")
    print(f"  Feature dimension: {features.shape[1]}")


if __name__ == "__main__":
    main()
