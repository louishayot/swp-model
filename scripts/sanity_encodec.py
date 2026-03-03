#!/usr/bin/env python3
"""EnCodec round-trip sanity check.

Generates a short synthetic waveform IN MEMORY, runs it through EnCodec,
computes all three metrics, prints a summary, and exits non-zero on failure.

No file I/O (no torchaudio.save).

Usage:
    python scripts/sanity_encodec.py
    python scripts/sanity_encodec.py --bandwidth 1.5
    python scripts/sanity_encodec.py --sr 44100 --duration 1.0
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Make swp importable when running as `python scripts/sanity_encodec.py` from repo root
_scripts_dir = os.path.dirname(os.path.realpath(__file__))
_repo_root = os.path.dirname(_scripts_dir)
sys.path.append(_repo_root)

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bandwidth", type=float, default=6.0,
        help="EnCodec bandwidth in kbps {1.5, 3.0, 6.0, 12.0, 24.0} (default: 6.0)"
    )
    p.add_argument(
        "--sr", type=int, default=16_000,
        help="Input sample rate for the synthetic waveform (default: 16000, tests resampling)"
    )
    p.add_argument(
        "--duration", type=float, default=0.5,
        help="Duration of the synthetic waveform in seconds (default: 0.5)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducible noise (default: 42)"
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Override device (e.g. cpu, cuda, mps). Default: auto-select."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Synthetic waveform (in-memory, no file I/O)
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    n_samples = int(args.sr * args.duration)
    wav = torch.randn(n_samples)  # white noise, shape [T]

    print(f"Input waveform:  {n_samples} samples @ {args.sr} Hz  ({args.duration:.2f}s)")
    print(f"Codec:           EnCodec (facebook/encodec_24khz), bandwidth={args.bandwidth} kbps")
    if args.device:
        print(f"Device override: {args.device}")
    print()

    # ------------------------------------------------------------------
    # 2. Load codec (imports transformers lazily)
    # ------------------------------------------------------------------
    print("Loading EnCodec …")
    kwargs: dict = {"bandwidth": args.bandwidth}
    if args.device:
        kwargs["device"] = args.device

    from swp.codecs import get_codec
    codec = get_codec("encodec", **kwargs)
    print(f"  device:        {codec.device}")
    print(f"  native SR:     {codec.sample_rate} Hz")
    print()

    # ------------------------------------------------------------------
    # 3. Reconstruct
    # ------------------------------------------------------------------
    print("Running reconstruct() …")
    result = codec.reconstruct(wav, sr=args.sr)
    print(f"  n_tokens:      {result['n_tokens']}")
    print(f"  tokens shape:  {tuple(result['tokens'].shape)}")  # [Q, T]
    print(f"  recon_sr:      {result['recon_sr']} Hz")
    print(f"  recon_wav len: {result['recon_wav'].shape[0]} samples")
    print()

    # ------------------------------------------------------------------
    # 4. Metrics (fully in-memory)
    # ------------------------------------------------------------------
    from swp.codecs.metrics import compute_all, is_valid

    print("Computing metrics …")
    metrics = compute_all(
        ref_wav=wav,
        result=result,
        codec=codec,
        ref_sr=args.sr,
    )

    print()
    print("─" * 40)
    print(f"  SI-SDR              : {metrics['si_sdr']:.3f} dB")
    print(f"  Mel distance        : {metrics['mel_distance']:.6f}")
    print(f"  Re-encode consistency: {metrics['reencode_consistency']:.4f}")
    print("─" * 40)

    # ------------------------------------------------------------------
    # 5. Sanity assertions
    # ------------------------------------------------------------------
    errors: list[str] = []

    if result["n_tokens"] <= 0:
        errors.append(f"n_tokens={result['n_tokens']} is not positive")

    if result["tokens"] is None:
        errors.append("tokens is None (expected [Q, T] Tensor)")
    else:
        if result["tokens"].dim() != 2:
            errors.append(f"tokens.dim()={result['tokens'].dim()}, expected 2 ([Q, T])")

    # SI-SDR: must be finite (negative is OK for white noise)
    sdr = metrics["si_sdr"]
    if math.isnan(sdr) or math.isinf(sdr):
        errors.append(f"SI-SDR is {sdr} (expected finite value)")

    # Mel distance: must be finite and non-negative
    mel_d = metrics["mel_distance"]
    if math.isnan(mel_d) or math.isinf(mel_d) or mel_d < 0:
        errors.append(f"mel_distance={mel_d:.6f} (expected finite non-negative value)")

    # Re-encode consistency: [0, 1] or NaN (NaN only if tokens unavailable)
    rc = metrics["reencode_consistency"]
    if math.isnan(rc):
        errors.append("reencode_consistency is NaN (tokens were expected to be available)")
    elif not (0.0 <= rc <= 1.0):
        errors.append(f"reencode_consistency={rc:.4f} outside [0, 1]")

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    print()
    if errors:
        print("[FAIL] Sanity check FAILED:")
        for e in errors:
            print(f"  ✗  {e}")
        sys.exit(1)
    else:
        print("[PASS] All sanity checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
