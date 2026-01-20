#!/usr/bin/env python3
"""Training script for acoustic models (Ua_w2v).

This script trains acoustic encoder models on pre-extracted wav2vec2/HuBERT
features. It mirrors the structure of train_repetition.py.

Usage:
    python scripts/train_acoustic.py \
        --manifest_path ./acoustic_features/manifest.json \
        --hidden_size 128 \
        --num_epochs 50 \
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

import torch.optim as optim

from swp.datasets.acoustic import get_acoustic_trainloader
from swp.models.acoustic_encoder import AcousticEncoder
from swp.models.autoencoder import Unimodel
from swp.models.decoders import DecoderLSTM, DecoderRNN
from swp.models.losses import AuditoryXENT, FirstErrorXENT
from swp.train.acoustic import train
from swp.utils.datasets import get_phoneme_to_id
from swp.utils.models import (
    AcousticArgs,
    get_model,
    get_model_name,
    get_model_name_from_args,
    get_train_args,
    get_train_name,
)
from swp.utils.setup import backend_setup, seed_everything, set_device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train acoustic encoder model on wav2vec2 features"
    )

    # Data arguments
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to manifest JSON file with acoustic features",
    )
    parser.add_argument(
        "--valid_manifest_path",
        type=str,
        default=None,
        help="Path to validation manifest (if different from train)",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name string, overrides other model parameters",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        default=None,
        help="Training name string, overrides other training parameters",
    )
    parser.add_argument(
        "--recur_type",
        type=str,
        default="lstm",
        help="Recurrent network architecture: RNN or LSTM",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Hidden size of recurrent subnetworks",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers in recurrent subnetworks",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=768,
        help="Input feature dimension (768 for wav2vec2-base, 1024 for large)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate for encoders and decoders",
    )
    parser.add_argument(
        "--tf_ratio",
        type=float,
        default=0.0,
        help="Teacher forcing ratio for decoder",
    )

    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learn_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="Maximum number of frames per sample (for memory)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="classic",
        help="Loss function to use: classic or first",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print logs during training",
    )

    args = parser.parse_args()

    # Setup
    seed_everything(args.seed)
    print(f"seed: {args.seed}")
    backend_setup()
    device = set_device()

    # Parse training name or build from args
    if args.train_name is None:
        batch_size = args.batch_size
        learn_rate = args.learn_rate
        include_stress = args.include_stress
        train_name = get_train_name(
            batch_size,
            learn_rate,
            fold_id=None,  # Acoustic doesn't use folds (for now)
            include_stress=include_stress,
            seed=args.seed,
            loss=args.loss,
        )
    else:
        train_name = args.train_name
        train_args = get_train_args(train_name)
        batch_size = train_args["batch_size"]
        learn_rate = train_args["learning_rate"]
        include_stress = train_args["include_stress"]

    # Build or load model
    if args.model_name is None:
        recur_type = args.recur_type.upper()
        if recur_type not in ["RNN", "LSTM"]:
            raise ValueError("Invalid recurrent layer type")
        Decoder = DecoderRNN if recur_type == "RNN" else DecoderLSTM

        phoneme_to_id = get_phoneme_to_id(include_stress)
        vocab_size = len(phoneme_to_id)

        # Create acoustic encoder
        encoder = AcousticEncoder(
            input_dim=args.input_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            recur_type=recur_type,
        )

        # Create decoder
        decoder = Decoder(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tf_ratio=args.tf_ratio,
        )

        # Create model
        model = Unimodel(encoder, decoder, start_token_id=phoneme_to_id["<SOS>"])
        model_name = get_model_name(model)
    else:
        model_name = args.model_name
        model = get_model(model_name)
        phoneme_to_id = get_phoneme_to_id(include_stress)

    # Create dataloaders
    manifest_path = Path(args.manifest_path)
    train_loader = get_acoustic_trainloader(
        manifest_path=manifest_path,
        batch_size=batch_size,
        include_stress=include_stress,
        max_frames=args.max_frames,
        shuffle=True,
    )

    # Use same manifest for validation if not specified
    valid_manifest_path = args.valid_manifest_path or args.manifest_path
    valid_loader = get_acoustic_trainloader(
        manifest_path=Path(valid_manifest_path),
        batch_size=batch_size,
        include_stress=include_stress,
        max_frames=args.max_frames,
        shuffle=False,
    )

    # Setup loss function
    if args.loss == "classic":
        criterion = AuditoryXENT()
    elif args.loss == "first":
        criterion = FirstErrorXENT()
    else:
        raise ValueError("Invalid loss function")

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    if args.verbose:
        print("-" * 60)
        print(f"\n{model_name}~{train_name}")
        print(f"Manifest: {manifest_path}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Device: {device}")
        print(f"Input dim: {args.input_dim}")

    # Train
    train(
        model=model,
        model_name=model_name,
        train_name=train_name,
        criterion=criterion,
        optimizer=optimizer,
        phoneme_to_id=phoneme_to_id,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=args.num_epochs,
        device=device,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"\n{model_name}~{train_name}\n")
        print("-" * 60)
