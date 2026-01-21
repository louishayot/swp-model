#!/usr/bin/env python3
"""Generate learning curve figures from acoustic model training logs.

Reads CSV logs from results/gridsearch/train/ and generates PNG figures.

By default, only generates figures for the primary runs (b10 overfit, b32 full).
Use --all to include legacy/experimental runs (b1, b8, etc.).

Usage:
    # Generate figures for primary runs (b10, b32)
    python scripts/make_acoustic_figures.py

    # Include all runs (including legacy b1, b8, etc.)
    python scripts/make_acoustic_figures.py --all

    # Generate figure for a specific model
    python scripts/make_acoustic_figures.py \
        --model_name Ua_w2v_LSTM_h128_l1_v42_d0.0_t0.0_s1__gi768 \
        --train_name b32_l0.001_fall_s42_sn_ec

    # Specify output directory
    python scripts/make_acoustic_figures.py --output_dir ./figs
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import pandas as pd


def get_train_log_dir() -> Path:
    """Get the directory containing training CSV logs."""
    return Path(parent) / "results" / "gridsearch" / "train"


def find_acoustic_logs(log_dir: Path) -> list[Path]:
    """Find all acoustic model training logs (Ua_w2v models)."""
    logs = []
    for csv_file in log_dir.glob("*.csv"):
        # Acoustic models have Ua_w2v in the name
        if "Ua_w2v" in csv_file.name:
            logs.append(csv_file)
    return sorted(logs)


def load_training_log(csv_path: Path) -> pd.DataFrame:
    """Load a training log CSV file."""
    df = pd.read_csv(csv_path, index_col=0)
    return df


def plot_learning_curve(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    show_errors: bool = True,
) -> None:
    """Plot learning curves (loss and optionally errors) from training log.

    Args:
        df: DataFrame with columns: Epoch, Train loss, Valid loss, Train errors, Valid errors
        title: Figure title
        output_path: Path to save the PNG
        show_errors: Whether to include error rate subplot
    """
    epochs = df["Epoch"].values
    train_loss = df["Train loss"].values
    valid_loss = df["Valid loss"].values

    if show_errors and "Train errors" in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss subplot
        ax1.plot(epochs, train_loss, "b-", label="Train loss", linewidth=1.5)
        ax1.plot(epochs, valid_loss, "r-", label="Valid loss", linewidth=1.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (Cross-Entropy)")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Error subplot
        train_errors = df["Train errors"].values
        valid_errors = df["Valid errors"].values
        ax2.plot(epochs, train_errors, "b-", label="Train errors", linewidth=1.5)
        ax2.plot(epochs, valid_errors, "r-", label="Valid errors", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Errors (count)")
        ax2.set_title("Errors")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss, "b-", label="Train loss", linewidth=1.5)
        ax.plot(epochs, valid_loss, "r-", label="Valid loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Cross-Entropy)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def extract_run_info(csv_name: str) -> tuple[str, str]:
    """Extract model_name and train_name from CSV filename.

    Format: {model_name}~{train_name}.csv
    """
    basename = csv_name.replace(".csv", "")
    parts = basename.split("~")
    if len(parts) == 2:
        return parts[0], parts[1]
    return basename, "unknown"


def make_short_label(model_name: str, train_name: str) -> str:
    """Create a short label for the figure title."""
    # Extract batch size from train_name (e.g., b32_l0.001_... -> b32)
    batch_part = train_name.split("_")[0] if "_" in train_name else train_name

    # Check if it's an overfit run (small batch size)
    if batch_part in ["b10", "b8", "b5"]:
        return f"Overfit ({batch_part})"
    else:
        return f"Full training ({batch_part})"


# Primary batch sizes from run_acoustic_smoke.sh
PRIMARY_BATCH_SIZES = {"b10", "b32"}


def is_primary_run(train_name: str) -> bool:
    """Check if this is a primary run (b10 overfit or b32 full training)."""
    batch_part = train_name.split("_")[0] if "_" in train_name else train_name
    return batch_part in PRIMARY_BATCH_SIZES


def main():
    parser = argparse.ArgumentParser(
        description="Generate learning curve figures from acoustic model training logs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Specific model name to plot (default: all acoustic models)",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        default=None,
        help="Specific train name to plot (requires --model_name)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./figs",
        help="Output directory for figures (default: ./figs)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory containing CSV logs (default: results/gridsearch/train)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include all runs (by default, only primary runs b10/b32 are included)",
    )

    args = parser.parse_args()

    # Set up directories
    log_dir = Path(args.log_dir) if args.log_dir else get_train_log_dir()
    output_dir = Path(args.output_dir)

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        print("Run training first to generate logs.")
        sys.exit(1)

    # Find logs to process
    if args.model_name and args.train_name:
        # Specific model
        csv_path = log_dir / f"{args.model_name}~{args.train_name}.csv"
        if not csv_path.exists():
            print(f"Error: Log file not found: {csv_path}")
            sys.exit(1)
        log_files = [csv_path]
    elif args.model_name:
        # All train runs for a specific model
        log_files = list(log_dir.glob(f"{args.model_name}~*.csv"))
        if not log_files:
            print(f"Error: No logs found for model: {args.model_name}")
            sys.exit(1)
    else:
        # All acoustic models
        log_files = find_acoustic_logs(log_dir)
        if not log_files:
            print(f"No acoustic model logs found in: {log_dir}")
            print("Run training first, or specify --model_name and --train_name")
            sys.exit(1)

    # Filter to primary runs unless --all is specified
    if not args.all and not (args.model_name and args.train_name):
        filtered_files = []
        skipped = []
        for csv_path in log_files:
            _, train_name = extract_run_info(csv_path.name)
            if is_primary_run(train_name):
                filtered_files.append(csv_path)
            else:
                skipped.append(csv_path.name)
        if skipped:
            print(f"Skipping {len(skipped)} legacy/experimental runs (use --all to include):")
            for name in skipped:
                print(f"  - {name}")
        log_files = filtered_files

    print(f"Found {len(log_files)} log file(s) to process")

    if not log_files:
        print("No matching logs found. Use --all to include legacy runs.")
        sys.exit(0)

    # Generate figures
    for csv_path in log_files:
        model_name, train_name = extract_run_info(csv_path.name)
        print(f"\nProcessing: {model_name}")
        print(f"  Train config: {train_name}")

        # Load data
        df = load_training_log(csv_path)
        print(f"  Epochs: {len(df)}")

        # Create figure title and filename
        short_label = make_short_label(model_name, train_name)
        title = f"Ua_w2v Learning Curve - {short_label}"

        # Safe filename (replace problematic characters)
        safe_name = f"{model_name}__{train_name}".replace(".", "_")
        output_path = output_dir / f"learning_curve_{safe_name}.png"

        # Plot
        plot_learning_curve(df, title, output_path)

    print(f"\nDone! Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
