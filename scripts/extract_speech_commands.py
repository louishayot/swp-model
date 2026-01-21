#!/usr/bin/env python3
"""Extract wav2vec2 features from Speech Commands dataset (word-level audio).

This script extracts acoustic features from the Speech Commands dataset,
which contains isolated single-word utterances - ideal for word repetition
experiments comparable to the Ua phoneme-input models.

Speech Commands v2 contains 35 words with ~100k utterances total:
- Core words: yes, no, up, down, left, right, on, off, stop, go
- Additional words: zero-nine, bed, bird, cat, dog, happy, house, etc.

Usage:
    python scripts/extract_speech_commands.py \
        --output_dir ./acoustic_features_words \
        --limit 1000

    # Overfit test on single word type:
    python scripts/extract_speech_commands.py \
        --output_dir ./acoustic_features_overfit \
        --words yes \
        --limit 10
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from pathlib import Path

import numpy as np
import torch
from nltk.corpus import cmudict
from tqdm import tqdm

from swp.utils.datasets import get_phoneme_to_id


# Mapping from Speech Commands word labels to their phoneme sequences
# Using CMU dictionary pronunciations (stress removed)
WORD_TO_PHONEMES = {
    # Digits
    "zero": ["Z", "IH", "R", "OW"],
    "one": ["W", "AH", "N"],
    "two": ["T", "UW"],
    "three": ["TH", "R", "IY"],
    "four": ["F", "AO", "R"],
    "five": ["F", "AY", "V"],
    "six": ["S", "IH", "K", "S"],
    "seven": ["S", "EH", "V", "AH", "N"],
    "eight": ["EY", "T"],
    "nine": ["N", "AY", "N"],
    # Commands
    "yes": ["Y", "EH", "S"],
    "no": ["N", "OW"],
    "up": ["AH", "P"],
    "down": ["D", "AW", "N"],
    "left": ["L", "EH", "F", "T"],
    "right": ["R", "AY", "T"],
    "on": ["AA", "N"],
    "off": ["AO", "F"],
    "stop": ["S", "T", "AA", "P"],
    "go": ["G", "OW"],
    # Additional words
    "bed": ["B", "EH", "D"],
    "bird": ["B", "ER", "D"],
    "cat": ["K", "AE", "T"],
    "dog": ["D", "AO", "G"],
    "happy": ["HH", "AE", "P", "IY"],
    "house": ["HH", "AW", "S"],
    "marvin": ["M", "AA", "R", "V", "AH", "N"],
    "sheila": ["SH", "IY", "L", "AH"],
    "tree": ["T", "R", "IY"],
    "wow": ["W", "AW"],
    "learn": ["L", "ER", "N"],
    "backward": ["B", "AE", "K", "W", "ER", "D"],
    "forward": ["F", "AO", "R", "W", "ER", "D"],
    "follow": ["F", "AA", "L", "OW"],
    "visual": ["V", "IH", "ZH", "UW", "AH", "L"],
}


def load_wav2vec2_model(model_name: str, device: torch.device):
    """Load wav2vec2 model and processor from HuggingFace."""
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return processor, model


def get_word_phonemes(word: str, phoneme_to_id: dict) -> list[str] | None:
    """Get phoneme sequence for a word, validating against vocabulary.

    Args:
        word: Word label from Speech Commands
        phoneme_to_id: Valid phoneme vocabulary

    Returns:
        List of phonemes if all are in vocabulary, None otherwise
    """
    # Try our predefined mapping first
    if word in WORD_TO_PHONEMES:
        phonemes = WORD_TO_PHONEMES[word]
        # Validate all phonemes are in vocabulary
        if all(p in phoneme_to_id for p in phonemes):
            return phonemes

    # Fallback to CMU dict
    try:
        cmu = cmudict.dict()
        if word.lower() in cmu:
            raw_phonemes = cmu[word.lower()][0]  # Take first pronunciation
            # Remove stress markers and validate
            phonemes = []
            for p in raw_phonemes:
                p_clean = p.rstrip("012")
                if p_clean in phoneme_to_id:
                    phonemes.append(p_clean)
                else:
                    return None  # Unknown phoneme
            return phonemes
    except LookupError:
        import nltk
        nltk.download('cmudict', quiet=True)
        return get_word_phonemes(word, phoneme_to_id)  # Retry after download

    return None


def extract_features_single(
    audio: np.ndarray,
    processor,
    model,
    device: torch.device,
    sampling_rate: int = 16000,
) -> np.ndarray:
    """Extract wav2vec2 features from a single audio sample.

    Args:
        audio: Audio array (1D, 16kHz)
        processor: Wav2Vec2Processor
        model: Wav2Vec2Model
        device: torch device
        sampling_rate: Audio sampling rate

    Returns:
        Feature array (time, feature_dim)
    """
    # Process audio
    inputs = processor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=False,
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        features = outputs.last_hidden_state[0]  # (time, hidden_dim)

    return features.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Extract wav2vec2 features from Speech Commands dataset"
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
        "--words",
        type=str,
        default=None,
        help="Comma-separated list of words to include (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per word (for testing)",
    )
    parser.add_argument(
        "--total_limit",
        type=int,
        default=None,
        help="Limit total number of samples across all words",
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
        help="Device (cuda, mps, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use",
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

    # Load wav2vec2 model
    print(f"Loading wav2vec2 model: {args.model}")
    processor, model = load_wav2vec2_model(args.model, device)

    # Load phoneme vocabulary
    phoneme_to_id = get_phoneme_to_id(args.include_stress)
    print(f"Phoneme vocabulary size: {len(phoneme_to_id)}")

    # Parse word filter
    word_filter = None
    if args.words:
        word_filter = set(w.strip().lower() for w in args.words.split(","))
        print(f"Filtering to words: {word_filter}")

    # Load Speech Commands dataset
    print(f"Loading Speech Commands dataset (split: {args.split})...")
    from datasets import load_dataset

    dataset = load_dataset(
        "google/speech_commands",
        "v0.02",
        split=args.split,
        trust_remote_code=True,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Get unique labels
    unique_labels = set(dataset["label"])
    # Map label indices to word names
    label_names = dataset.features["label"].names
    print(f"Available words ({len(label_names)}): {label_names[:10]}...")

    # Build word -> phonemes mapping and validate
    valid_words = {}
    for label_idx, word in enumerate(label_names):
        if word_filter and word.lower() not in word_filter:
            continue
        phonemes = get_word_phonemes(word, phoneme_to_id)
        if phonemes:
            valid_words[label_idx] = {
                "word": word,
                "phonemes": phonemes,
            }
        else:
            print(f"  Skipping '{word}' - phonemes not in vocabulary")

    print(f"Valid words for extraction: {len(valid_words)}")
    for label_idx, info in list(valid_words.items())[:5]:
        print(f"  {info['word']}: {' '.join(info['phonemes'])}")

    # Process samples
    manifest = {}
    feature_dir = output_dir / "features"
    feature_dir.mkdir(exist_ok=True)

    # Track counts per word for limiting
    word_counts = {label_idx: 0 for label_idx in valid_words}
    total_count = 0
    skipped_unknown = 0
    skipped_limit = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Extracting features")):
        label_idx = sample["label"]

        # Skip unknown words
        if label_idx not in valid_words:
            skipped_unknown += 1
            continue

        word_info = valid_words[label_idx]
        word = word_info["word"]

        # Check per-word limit
        if args.limit and word_counts[label_idx] >= args.limit:
            skipped_limit += 1
            continue

        # Check total limit
        if args.total_limit and total_count >= args.total_limit:
            break

        # Get audio
        audio = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]

        # Resample if needed (wav2vec2 expects 16kHz)
        if sampling_rate != 16000:
            import torchaudio.functional as F
            audio_tensor = torch.from_numpy(audio).float()
            audio = F.resample(audio_tensor, sampling_rate, 16000).numpy()

        # Extract features
        features = extract_features_single(audio, processor, model, device)

        # Create utterance ID
        utterance_id = f"{word}_{idx:06d}"

        # Save features
        feature_path = feature_dir / f"{utterance_id}.npz"
        np.savez_compressed(feature_path, features=features)

        # Add to manifest
        manifest[utterance_id] = {
            "feature_path": f"features/{utterance_id}.npz",
            "phonemes": word_info["phonemes"],
            "num_frames": features.shape[0],
            "word": word,
        }

        word_counts[label_idx] += 1
        total_count += 1

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print(f"\nExtraction complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Total samples: {len(manifest)}")
    if len(manifest) > 0:
        sample_features = np.load(feature_dir / f"{list(manifest.keys())[0]}.npz")
        print(f"  Feature dimension: {sample_features['features'].shape[1]}")

    print(f"\nPer-word counts:")
    for label_idx, count in word_counts.items():
        if count > 0:
            print(f"  {valid_words[label_idx]['word']}: {count}")

    print(f"\nSkipped: {skipped_unknown} (unknown words), {skipped_limit} (limit reached)")


if __name__ == "__main__":
    main()
