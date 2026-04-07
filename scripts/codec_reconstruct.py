#!/usr/bin/env python3
"""Codec-agnostic reconstruction pipeline.

For each audio file in a dataset CSV, loads the waveform, runs it through
a codec, and computes reconstruction metrics. Results are cached as a CSV
alongside a JSON sidecar that records all parameters.

Metrics comparison policy:
  The reference waveform is resampled to the codec's native sample rate
  before computing SI-SDR and mel-distance. This is handled internally by
  swp.codecs.metrics.compute_all(). Both signals are truncated to the shorter
  of the two before any metric computation.

Usage:
    python scripts/codec_reconstruct.py \\
        --codec encodec --codec-arg bandwidth=6.0 \\
        --dataset data/external/mald/subset_tiny.csv

    python scripts/codec_reconstruct.py \\
        --codec encodec --codec-arg bandwidth=1.5 \\
        --dataset data/external/mald/subset_tiny.csv \\
        --regenerate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_scripts_dir = os.path.dirname(os.path.realpath(__file__))
_repo_root = os.path.dirname(_scripts_dir)
sys.path.append(_repo_root)

import numpy as np
import pandas as pd
import torch
import torchaudio

from swp.codecs import get_codec
from swp.codecs.metrics import compute_all
from swp.utils.paths import get_codec_results_dir

# ---------------------------------------------------------------------------
# Silence utilities
# ---------------------------------------------------------------------------

_SILENCE_FRAME = 512      # samples per frame for RMS computation
_SILENCE_THRESH = 1e-4    # RMS threshold below which a frame is "silent"


def compute_silence_ratio(wav: torch.Tensor) -> float:
    """Fraction of non-overlapping frames whose RMS falls below threshold.

    Args:
        wav: 1-D waveform tensor.

    Returns:
        float in [0.0, 1.0]; 0.0 if the waveform is shorter than one frame.
    """
    n_frames = len(wav) // _SILENCE_FRAME
    if n_frames == 0:
        return 0.0
    frames = wav[: n_frames * _SILENCE_FRAME].reshape(n_frames, _SILENCE_FRAME)
    rms = frames.pow(2).mean(dim=1).sqrt()
    return (rms < _SILENCE_THRESH).float().mean().item()


def trim_silence(wav: torch.Tensor) -> torch.Tensor:
    """Remove leading and trailing silent frames from a 1-D waveform.

    If the entire waveform is silent, returns the original (untrimmed).
    """
    n_frames = len(wav) // _SILENCE_FRAME
    if n_frames == 0:
        return wav
    frames_part = wav[: n_frames * _SILENCE_FRAME].reshape(n_frames, _SILENCE_FRAME)
    rms = frames_part.pow(2).mean(dim=1).sqrt()
    voiced_idx = (rms >= _SILENCE_THRESH).nonzero(as_tuple=True)[0]
    if len(voiced_idx) == 0:
        return wav  # all silent — return as-is
    first = int(voiced_idx[0].item()) * _SILENCE_FRAME
    last = min(int(voiced_idx[-1].item() + 1) * _SILENCE_FRAME, len(wav))
    return wav[first:last]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_codec_arg(s: str):
    """Parse 'key=value'; value is cast to int, then float, then left as str."""
    key, _, val_str = s.partition("=")
    key = key.strip()
    val_str = val_str.strip()
    for cast in (int, float):
        try:
            return key, cast(val_str)
        except ValueError:
            pass
    return key, val_str


def _compute_paramhash(codec_kwargs: dict) -> str:
    """Deterministic 8-character MD5 hash of sorted codec kwargs."""
    serialized = json.dumps(codec_kwargs, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--codec", required=True,
                   help="Codec name (e.g. 'encodec'). Must be registered in swp.codecs.")
    p.add_argument("--codec-arg", action="append", default=[], dest="codec_args",
                   metavar="KEY=VALUE",
                   help="Codec constructor keyword argument (repeatable). "
                        "Values are auto-cast to int/float/str.")
    p.add_argument("--dataset", required=True,
                   help="Input CSV with at least an 'audio_path' column.")
    p.add_argument("--output", default=None,
                   help="Override output CSV path. Default: auto-generated from codec+dataset.")
    p.add_argument("--trim-silence", action="store_true", dest="trim_silence",
                   help="Trim leading/trailing silence before passing audio to codec.")
    p.add_argument("--regenerate", action="store_true",
                   help="Overwrite cached output even if it already exists.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Parse codec kwargs
    codec_kwargs: dict = {}
    for s in args.codec_args:
        k, v = _parse_codec_arg(s)
        codec_kwargs[k] = v

    # Safety: warn if silence trimming is requested without an explicit output path.
    # The paramhash is derived from codec kwargs only (does not include trim_silence),
    # so a trimmed run at bw=6.0 produces the same auto-path as a non-trimmed run.
    if args.trim_silence and args.output is None:
        print(
            "WARNING: --trim-silence is set but --output was not specified.\n"
            "  The auto-generated filename hashes codec kwargs only (ignores the trim flag).\n"
            "  A non-trimmed run at the same bandwidth will be overwritten.\n"
            "  Recommended: add --output reproduce/data/codec/encodec/subset__bw6_TRIM.csv"
        )

    # Resolve output paths
    paramhash = _compute_paramhash(codec_kwargs)
    dataset_stem = Path(args.dataset).stem   # e.g. "subset_tiny"

    if args.output:
        out_csv = Path(args.output)
        out_meta = out_csv.with_suffix(".meta.json")
    else:
        results_dir = get_codec_results_dir(args.codec)
        fname = f"{dataset_stem}__{paramhash}"
        out_csv = results_dir / f"{fname}.csv"
        out_meta = results_dir / f"{fname}.meta.json"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Caching check
    if out_csv.exists() and not args.regenerate:
        print(f"Cached results found: {out_csv}")
        print("Use --regenerate to overwrite.")
        return

    print(f"Codec       : {args.codec}  params={codec_kwargs}")
    print(f"Dataset     : {args.dataset}")
    print(f"Output CSV  : {out_csv}")
    print(f"Trim silence: {args.trim_silence}")
    print()

    # Load dataset
    df_in = pd.read_csv(args.dataset)
    if "audio_path" not in df_in.columns:
        print("ERROR: input CSV must contain an 'audio_path' column.", file=sys.stderr)
        sys.exit(1)

    # Load codec (downloads weights on first use)
    print(f"Loading codec '{args.codec}' …")
    codec = get_codec(args.codec, **codec_kwargs)
    print(f"  device     : {codec.device}")
    print(f"  native SR  : {codec.sample_rate} Hz")
    print()

    # Metadata columns to carry through from input CSV
    meta_cols = [c for c in df_in.columns if c != "audio_path"]

    # ---------------------------------------------------------------------------
    # Per-item processing
    # ---------------------------------------------------------------------------
    rows: list[dict] = []
    n_errors = 0
    _torchcodec_missing = False  # set True on first torchcodec-related ImportError

    for idx, row in df_in.iterrows():
        audio_path = str(row["audio_path"])
        word = row.get("word", audio_path)

        # Carry-through metadata
        record: dict = {c: row[c] for c in meta_cols if c in row.index}

        if not audio_path or not Path(audio_path).exists():
            print(f"  [{idx:>3}] SKIP  '{word}' — audio_path not found: {audio_path}")
            record.update(_nan_metrics(args.codec, codec_kwargs, args.trim_silence))
            rows.append(record)
            continue

        if _torchcodec_missing:
            n_errors += 1
            record.update(_nan_metrics(args.codec, codec_kwargs, args.trim_silence))
            rows.append(record)
            continue

        try:
            # -------- Load waveform --------
            waveform, sr = torchaudio.load(audio_path)  # [C, T]

            # Mono: average channels if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)  # [T]
            else:
                waveform = waveform.squeeze(0)   # [T]

            # -------- Silence metrics (always on raw waveform) --------
            silence_ratio = compute_silence_ratio(waveform)
            duration_pretrim_s = waveform.shape[0] / sr

            # -------- Optional silence trimming --------
            if args.trim_silence:
                waveform = trim_silence(waveform)

            duration_s = waveform.shape[0] / sr

            # Sanity: cross-check vs metadata duration if available
            dur_meta = row.get("duration_meta_s", None)
            if dur_meta is not None and not np.isnan(float(dur_meta)):
                discrepancy = abs(duration_pretrim_s - float(dur_meta))
                if discrepancy > 0.05:
                    print(
                        f"  [{idx:>3}] NOTE  '{word}': measured duration "
                        f"{duration_pretrim_s:.3f}s vs metadata {float(dur_meta):.3f}s "
                        f"(Δ={discrepancy:.3f}s)"
                    )

            if silence_ratio > 0.3:
                print(f"  [{idx:>3}] WARN  '{word}': high silence_ratio={silence_ratio:.2f}")

            # -------- Codec round-trip --------
            result = codec.reconstruct(waveform, sr)

            # -------- Metrics --------
            metrics = compute_all(
                ref_wav=waveform,
                result=result,
                codec=codec,
                ref_sr=sr,
            )

            record.update(
                duration_s=duration_s,
                duration_meta_s=dur_meta if dur_meta is not None else np.nan,
                silence_ratio=silence_ratio,
                codec_name=args.codec,
                codec_params=json.dumps(codec_kwargs, sort_keys=True),
                trim_silence=args.trim_silence,
                n_tokens=result["n_tokens"],
                si_sdr=metrics["si_sdr"],
                mel_distance=metrics["mel_distance"],
                reencode_consistency=metrics["reencode_consistency"],
            )
            if args.trim_silence:
                record["duration_pretrim_s"] = duration_pretrim_s

            sdr_str = f"{metrics['si_sdr']:.2f}" if not (
                metrics['si_sdr'] != metrics['si_sdr']  # nan
                or metrics['si_sdr'] in (float("inf"), float("-inf"))
            ) else str(metrics['si_sdr'])
            print(
                f"  [{idx:>3}] OK    '{word}'  "
                f"dur={duration_s:.2f}s  "
                f"n_tok={result['n_tokens']}  "
                f"SI-SDR={sdr_str}  "
                f"mel={metrics['mel_distance']:.4f}  "
                f"re-enc={metrics['reencode_consistency']:.3f}"
            )

        except Exception as exc:  # noqa: BLE001
            n_errors += 1
            if isinstance(exc, ImportError) and not _torchcodec_missing:
                _torchcodec_missing = True
                print(
                    f"  [{idx:>3}] FATAL  '{word}': {exc}\n"
                    f"  torchaudio.load() requires torchcodec — "
                    f"install it: pip install torchcodec==0.1  "
                    f"(or: pip install -r requirements.txt)\n"
                    f"  Marking all remaining items as failed."
                )
            else:
                print(f"  [{idx:>3}] ERROR '{word}': {exc}")
                traceback.print_exc()
            record.update(_nan_metrics(args.codec, codec_kwargs, args.trim_silence))

        rows.append(record)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)

    # Sidecar meta JSON
    meta = {
        "codec": args.codec,
        "params": codec_kwargs,
        "dataset": args.dataset,
        "dataset_stem": dataset_stem,
        "paramhash": paramhash,
        "trim_silence": args.trim_silence,
        "n_items": len(df_out),
        "n_errors": n_errors,
        "metrics_policy": (
            "ref resampled to codec native SR for SI-SDR and mel-distance comparison"
        ),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"Results saved : {out_csv}  ({len(df_out)} rows, {n_errors} errors)")
    print(f"Meta saved    : {out_meta}")
    if n_errors:
        print(f"WARNING: {n_errors} item(s) failed — check stderr above.")

    # Quick distribution summary
    if "mel_distance" in df_out.columns:
        valid = df_out["mel_distance"].dropna()
        if len(valid):
            print(
                f"\nmel_distance  : mean={valid.mean():.4f}  "
                f"median={valid.median():.4f}  SD={valid.std():.4f}  "
                f"range=[{valid.min():.4f}, {valid.max():.4f}]"
            )
    if "si_sdr" in df_out.columns:
        valid = df_out["si_sdr"].replace([float("inf"), float("-inf")], np.nan).dropna()
        if len(valid):
            print(
                f"SI-SDR (dB)   : mean={valid.mean():.2f}  "
                f"median={valid.median():.2f}  SD={valid.std():.2f}"
            )
    if "reencode_consistency" in df_out.columns:
        valid = df_out["reencode_consistency"].dropna()
        if len(valid):
            print(f"reencode_cons : mean={valid.mean():.4f}")


def _nan_metrics(codec_name: str, codec_kwargs: dict, trim_silence: bool) -> dict:
    """Return NaN-filled metric dict for failed items."""
    d = dict(
        duration_s=np.nan,
        duration_meta_s=np.nan,
        silence_ratio=np.nan,
        codec_name=codec_name,
        codec_params=json.dumps(codec_kwargs, sort_keys=True),
        trim_silence=trim_silence,
        n_tokens=np.nan,
        si_sdr=np.nan,
        mel_distance=np.nan,
        reencode_consistency=np.nan,
    )
    if trim_silence:
        d["duration_pretrim_s"] = np.nan
    return d


if __name__ == "__main__":
    main()
