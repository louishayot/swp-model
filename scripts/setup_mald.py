#!/usr/bin/env python3
"""Parse MALD metadata and create a balanced subset for codec evaluation.

Assumes MALD data has been manually downloaded to data/external/mald/:
  - MALD1_1_ItemData.txt  (tab-delimited item-level metadata)
  - words/                (word WAV files, one per item)
  - pseudowords/          (pseudoword WAV files, one per item)

Downloads:
  ItemData : https://osf.io/download/...  (MALD1_1_ItemData.txt, ~6.7 MB)
  Word audio     : https://doi.org/10.7939/r3-v0jr-rr12
  Pseudoword audio: https://doi.org/10.7939/r3-v7jh-p314

Usage:
    python scripts/setup_mald.py --tiny    # 20 items: 10 words + 10 pseudowords
    python scripts/setup_mald.py           # 400 items: 200 words + 200 pseudowords
"""

from __future__ import annotations

import argparse
import os
import sys

_scripts_dir = os.path.dirname(os.path.realpath(__file__))
_repo_root = os.path.dirname(_scripts_dir)
sys.path.append(_repo_root)

import pathlib
from io import StringIO

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = pathlib.Path(_repo_root) / "data" / "external" / "mald"

# Priority-ordered candidate column names (first match wins)
_COL_CANDIDATES: dict[str, list[str]] = {
    "word": ["Item", "Word", "word", "item", "ITEM"],
    "is_word": ["IsWord", "is_word", "isword", "Lexicality", "lexicality"],
    "num_phones": ["NumPhones", "NPhones", "Nphones", "num_phones", "nphones", "PhoneCount"],
    "num_sylls": ["NumSylls", "NSylls", "Nsylls", "num_sylls", "nsylls", "SyllCount"],
    "duration_meta_s": ["Duration", "duration", "Dur", "dur"],
}

_SPEAKER_CANDIDATES = [
    "Speaker", "Talker", "Session", "List", "speaker", "talker", "session",
    "list_id", "TalkerID", "RecordingSession",
]

_FREQ_CANDIDATES = [
    "SUBTLWF", "Zipf_SUBTLEX", "Zipf", "Freq_pm", "LogFreq",
    "SUBTLEX", "SUBTLfreq", "Log_Freq_HAL", "Freq_HAL",
    "BiphonProb", "UniphonProb",
]

# Exact MALD column names included verbatim when present.
# Numeric columns are coerced to float; string columns kept as-is.
_EXTRA_MALD_COLS = [
    "PhonotacticProbability",
    "StressPattern",
    "TempUP", "OrthUP", "PhonUP",
    "FreqSUBTLEX", "FreqCOCA", "FreqCOCAspok", "FreqGoogle",
]
_EXTRA_MALD_COLS_STR = {"StressPattern"}  # non-numeric; kept as string


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_audio_path(word: str, is_word: bool, data_dir: pathlib.Path) -> str | None:
    """Try common MALD audio naming patterns. Returns None if not found."""
    subdir = "words" if is_word else "pseudowords"
    base = data_dir / subdir
    if not base.exists():
        return None

    # Try exact name and common case variants
    for stem in [word, word.lower(), word.upper(), word.capitalize()]:
        for ext in [".wav", ".WAV"]:
            p = base / f"{stem}{ext}"
            if p.exists():
                return str(p)

    # Case-insensitive scan (slower, fallback)
    word_lower = word.lower()
    try:
        for p in base.iterdir():
            if p.stem.lower() == word_lower and p.suffix.lower() == ".wav":
                return str(p)
    except PermissionError:
        pass

    return None


def _sample_stratified(
    group_df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    label: str,
    log_fn,
) -> pd.DataFrame:
    """Proportional stratified sampling by num_phones (rounded to int).

    Returns exactly min(n, len(group_df)) rows.
    """
    group_df = group_df.copy().dropna(subset=["num_phones"])
    group_df["_np_int"] = group_df["num_phones"].round().astype(int)

    counts = group_df["_np_int"].value_counts().sort_index()
    total = len(group_df)

    if total <= n:
        log_fn(f"    {label}: only {total} items available (wanted {n}); taking all")
        return group_df.drop(columns=["_np_int"])

    # Proportional targets, minimum 1 per bin
    targets: dict[int, int] = {
        b: max(1, int(round(n * c / total))) for b, c in counts.items()
    }

    # Adjust to exactly n (single-pass, no infinite-loop risk)
    allocated = sum(targets.values())
    bins_by_size_desc = sorted(counts.index, key=lambda x: -counts[x])

    # Reduce: remove 1 from the most-populated bins first, keeping min=1
    for b in bins_by_size_desc:
        if allocated <= n:
            break
        if targets[b] > 1:
            targets[b] -= 1
            allocated -= 1

    # Grow: add 1 to the bins with the most remaining room
    bins_by_room = sorted(counts.index, key=lambda x: -(counts[x] - targets[x]))
    for b in bins_by_room:
        if allocated >= n:
            break
        if targets[b] < counts[b]:
            targets[b] += 1
            allocated += 1

    # Sample from each bin
    parts = []
    for b, target in sorted(targets.items()):
        bin_items = group_df[group_df["_np_int"] == b]
        take = min(target, len(bin_items))
        seed_val = int(rng.integers(1_000_000))
        sampled = bin_items.sample(n=take, random_state=seed_val)
        parts.append(sampled)
        if take < target:
            log_fn(f"    bin num_phones={b}: wanted {target}, only {take} available")

    result = pd.concat(parts).drop(columns=["_np_int"])

    # Fill up if still short (shouldn't happen, but be safe)
    if len(result) < n:
        remaining = group_df[~group_df.index.isin(result.index)].drop(columns=["_np_int"])
        extra_n = n - len(result)
        if len(remaining) >= extra_n:
            extra = remaining.sample(n=extra_n, random_state=int(rng.integers(1_000_000)))
            result = pd.concat([result, extra])
            log_fn(f"    {label}: filled {extra_n} extra items from remaining pool")

    return result.head(n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tiny", action="store_true",
        help="Generate tiny subset (20 items: 10 words + 10 pseudowords)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
        dest="data_dir",
        help=f"Directory containing MALD data (default: {_DEFAULT_DATA_DIR})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir
    metadata_file = data_dir / "MALD1_1_ItemData.txt"

    if not metadata_file.exists():
        print(f"ERROR: Metadata file not found: {metadata_file}", file=sys.stderr)
        print(
            "Download MALD1_1_ItemData.txt and place it in that directory.\n"
            "See: https://doi.org/10.7939/r3-v0jr-rr12",
            file=sys.stderr,
        )
        sys.exit(1)

    # ----------------------------------------------------------------
    # Diagnostics buffer (mirrors to stdout + file)
    # ----------------------------------------------------------------
    diag = StringIO()

    def log(msg: str = "") -> None:
        print(msg)
        print(msg, file=diag)

    log("=" * 60)
    log("MALD METADATA DIAGNOSTICS")
    log("=" * 60)
    log()
    log(f"Metadata file : {metadata_file}")

    # ----------------------------------------------------------------
    # Load raw metadata
    # ----------------------------------------------------------------
    df_raw = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    log(f"Rows: {len(df_raw):,}  |  Columns: {len(df_raw.columns)}")
    log()
    log("ALL COLUMNS:")
    for c in df_raw.columns:
        log(f"  {c}")

    # ----------------------------------------------------------------
    # Map columns to canonical names
    # ----------------------------------------------------------------
    log()
    log("COLUMN MAPPING:")
    col_map: dict[str, str] = {}
    for canon, candidates in _COL_CANDIDATES.items():
        found = _find_col(df_raw, candidates)
        if found:
            col_map[canon] = found
            log(f"  {canon:20s} → '{found}'")
        else:
            log(f"  {canon:20s} → NOT FOUND  (tried: {candidates})")

    missing_required = [k for k in ("word", "is_word", "num_phones") if k not in col_map]
    if missing_required:
        log()
        log(f"ERROR: Required columns not found: {missing_required}")
        _save_diagnostics(data_dir, diag)
        sys.exit(1)

    # ----------------------------------------------------------------
    # Speaker / session columns
    # ----------------------------------------------------------------
    log()
    log("SPEAKER / SESSION COLUMNS:")
    speaker_cols_found = []
    for c in _SPEAKER_CANDIDATES:
        if c in df_raw.columns:
            speaker_cols_found.append(c)
            unique_vals = sorted(df_raw[c].dropna().unique())
            preview = unique_vals[:10]
            suffix = "..." if len(unique_vals) > 10 else ""
            log(f"  '{c}': {len(unique_vals)} unique values → {preview}{suffix}")
    if not speaker_cols_found:
        log(
            "  None found. MALD is documented as single-talker (confirmed by "
            "absence of speaker/session columns)."
        )

    # ----------------------------------------------------------------
    # Frequency columns
    # ----------------------------------------------------------------
    freq_cols_found = [c for c in _FREQ_CANDIDATES if c in df_raw.columns]
    log()
    log(f"FREQUENCY COLUMNS FOUND: {freq_cols_found}")
    extra_mald_found = [c for c in _EXTRA_MALD_COLS if c in df_raw.columns]
    log(f"EXTRA MALD COLUMNS FOUND: {extra_mald_found}")

    # ----------------------------------------------------------------
    # Build canonical DataFrame
    # ----------------------------------------------------------------
    df = pd.DataFrame()
    df["word"] = df_raw[col_map["word"]].astype(str).str.strip()

    # is_word: handle TRUE/FALSE strings and 1/0 integers
    raw_isword = df_raw[col_map["is_word"]]
    if raw_isword.dtype == object:
        df["is_word"] = (
            raw_isword.str.strip().str.upper()
            .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
        )
    else:
        df["is_word"] = raw_isword.astype(bool)

    df["num_phones"] = pd.to_numeric(df_raw[col_map["num_phones"]], errors="coerce")

    if "num_sylls" in col_map:
        df["num_sylls"] = pd.to_numeric(df_raw[col_map["num_sylls"]], errors="coerce")
    else:
        df["num_sylls"] = np.nan

    if "duration_meta_s" in col_map:
        dur_raw = pd.to_numeric(df_raw[col_map["duration_meta_s"]], errors="coerce")
        median_dur = dur_raw.median()
        if median_dur > 10:
            log()
            log(
                f"  Duration column '{col_map['duration_meta_s']}' looks like ms "
                f"(median={median_dur:.1f}); converting to seconds."
            )
            df["duration_meta_s"] = dur_raw / 1000.0
        else:
            df["duration_meta_s"] = dur_raw
    else:
        df["duration_meta_s"] = np.nan

    for fc in freq_cols_found:
        df[fc] = pd.to_numeric(df_raw[fc], errors="coerce")

    for col in extra_mald_found:
        if col in _EXTRA_MALD_COLS_STR:
            df[col] = df_raw[col].astype(str)
        else:
            df[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Drop rows missing required fields
    n_before = len(df)
    df = df.dropna(subset=["word", "is_word", "num_phones"]).reset_index(drop=True)
    if len(df) < n_before:
        log()
        log(f"Dropped {n_before - len(df)} rows with missing word/is_word/num_phones.")

    df.insert(0, "item_id", df.index)

    # ----------------------------------------------------------------
    # Summary statistics by lexicality
    # ----------------------------------------------------------------
    log()
    log("SUMMARY STATISTICS BY LEXICALITY (full dataset):")
    for is_w, label in [(True, "Words"), (False, "Pseudowords")]:
        sub = df[df["is_word"] == is_w]
        log(f"\n  {label} (n={len(sub):,}):")
        if not sub["num_phones"].isna().all():
            log(
                f"    num_phones  : mean={sub['num_phones'].mean():.2f}, "
                f"median={sub['num_phones'].median():.1f}, "
                f"range=[{sub['num_phones'].min():.0f}, {sub['num_phones'].max():.0f}]"
            )
        if "duration_meta_s" in sub and not sub["duration_meta_s"].isna().all():
            log(
                f"    duration_s  : mean={sub['duration_meta_s'].mean():.3f}, "
                f"median={sub['duration_meta_s'].median():.3f}, "
                f"SD={sub['duration_meta_s'].std():.3f}"
            )

    # ----------------------------------------------------------------
    # Stratified subset selection
    # ----------------------------------------------------------------
    n_per_lex = 10 if args.tiny else 200
    subset_name = "subset_tiny" if args.tiny else "subset"
    rng = np.random.default_rng(args.seed)

    log()
    log(f"SUBSET SELECTION ({'TINY' if args.tiny else 'FULL'}: "
        f"{n_per_lex} per lexicality, seed={args.seed}):")

    words_df = df[df["is_word"] == True]
    pseudo_df = df[df["is_word"] == False]

    words_subset = _sample_stratified(words_df, n_per_lex, rng, "Words", log)
    pseudo_subset = _sample_stratified(pseudo_df, n_per_lex, rng, "Pseudowords", log)

    subset = pd.concat([words_subset, pseudo_subset]).reset_index(drop=True)
    subset["item_id"] = range(len(subset))

    log()
    log(f"  Selected: {(subset['is_word'] == True).sum()} words + "
        f"{(subset['is_word'] == False).sum()} pseudowords = {len(subset)} total")
    log()
    log("  num_phones distribution in subset:")
    for is_w, label in [(True, "Words"), (False, "Pseudowords")]:
        sub = subset[subset["is_word"] == is_w]
        counts = sub["num_phones"].value_counts().sort_index()
        log(f"    {label}: {dict(counts.astype(int))}")

    if "duration_meta_s" in subset.columns and not subset["duration_meta_s"].isna().all():
        log()
        log("  duration_meta_s distribution in subset:")
        for is_w, label in [(True, "Words"), (False, "Pseudowords")]:
            sub = subset[subset["is_word"] == is_w]
            d = sub["duration_meta_s"].dropna()
            if len(d):
                log(f"    {label}: mean={d.mean():.3f}s, median={d.median():.3f}s, "
                    f"SD={d.std():.3f}s")

    # ----------------------------------------------------------------
    # Audio path discovery
    # ----------------------------------------------------------------
    log()
    log("AUDIO PATH DISCOVERY:")
    audio_paths: list[str] = []
    missing_audio: list[str] = []

    for _, row in subset.iterrows():
        path = _find_audio_path(str(row["word"]), bool(row["is_word"]), data_dir)
        audio_paths.append(path if path else "")
        if path is None:
            missing_audio.append(str(row["word"]))

    subset["audio_path"] = audio_paths
    found_n = sum(1 for p in audio_paths if p)
    log(f"  Found audio for {found_n}/{len(subset)} items")

    if missing_audio:
        preview = missing_audio[:10]
        suffix = "..." if len(missing_audio) > 10 else ""
        log(f"  Missing ({len(missing_audio)}): {preview}{suffix}")
        if len(missing_audio) > len(subset) * 0.5:
            log(
                f"  WARNING: >50% of audio files are missing. Ensure WAV files are in\n"
                f"  {data_dir}/words/  and  {data_dir}/pseudowords/"
            )

    # ----------------------------------------------------------------
    # Save subset CSV
    # ----------------------------------------------------------------
    base_cols = ["item_id", "word", "is_word", "num_phones", "num_sylls",
                 "duration_meta_s", "audio_path"]
    extra_cols = [c for c in freq_cols_found + extra_mald_found if c in subset.columns]
    final_cols = [c for c in base_cols + extra_cols if c in subset.columns]
    subset_out = subset[final_cols]

    output_path = data_dir / f"{subset_name}.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    subset_out.to_csv(output_path, index=False)

    log()
    log(f"Saved subset CSV  : {output_path}")
    log(f"Schema            : {list(subset_out.columns)}")

    # ----------------------------------------------------------------
    # Save diagnostics
    # ----------------------------------------------------------------
    _save_diagnostics(data_dir, diag)


def _save_diagnostics(data_dir: pathlib.Path, diag: StringIO) -> None:
    diag_path = data_dir / "subset_diagnostics.txt"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        f.write(diag.getvalue())
    print(f"Diagnostics saved : {diag_path}")


if __name__ == "__main__":
    main()
