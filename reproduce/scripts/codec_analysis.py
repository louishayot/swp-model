#!/usr/bin/env python3
"""Codec reconstruction analysis: figures and regression.

Reads one or more results CSVs produced by scripts/codec_reconstruct.py,
generates per-codec figures, and (when multiple CSVs are provided) cross-codec
comparison figures.

Per-codec figures go into a directory that is unique per codec+params:
  reproduce/figures/codec/{codec_name}/{params_str}/
  e.g. reproduce/figures/codec/encodec/bandwidth=6.0/
       reproduce/figures/codec/encodec/bandwidth=1.5/

Files inside each per-condition directory:
  error_vs_num_phones.png      – primary: mel distance vs phoneme count
  error_vs_duration.png        – control: mel distance vs audio duration (binned)
  tokens_vs_num_phones.png     – codec token count vs phoneme count
  regression_lexicality.png    – OLS coefficient plot
  regression_summary.txt       – full OLS summary table

Cross-codec / summary outputs (reproduce/figures/codec/comparison/):
  error_vs_num_phones_overlay.png  – overlay codecs [2+ CSVs]
  regression_comparison.png        – OLS coefficient bar chart [2+ CSVs]
  summary_table.csv                – per (codec, lexicality, num_phones) means
  length_effect_slopes.csv         – OLS slope of mel_distance on num_phones

Usage:
    # Single codec
    python reproduce/scripts/codec_analysis.py \\
        --results reproduce/data/codec/encodec/subset__<hash>.csv

    # Compare two bandwidth conditions, CCN phone range, global filter
    python reproduce/scripts/codec_analysis.py \\
        --results reproduce/data/codec/encodec/subset__<h1>.csv \\
                  reproduce/data/codec/encodec/subset__<h2>.csv \\
        --phones-min 3 --phones-max 9 --filter-all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# sys.path: make swp importable (script lives in reproduce/scripts/)
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))   # reproduce/scripts/
_repo_root = os.path.dirname(os.path.dirname(_this_dir))  # repo root
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pathlib

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\).*",
)

sns.set_palette("colorblind")

from swp.utils.paths import get_codec_figures_dir, get_codec_comparison_figures_dir

# ---------------------------------------------------------------------------
# Style constants — match swp/viz/test/length.py conventions
# ---------------------------------------------------------------------------
_LEX_PALETTE = {"word": "red", "pseudo": "blue"}
_LEX_LABELS = {True: "word", False: "pseudo", 1: "word", 0: "pseudo"}
_MARKER_SIZE = 8
_LINE_WIDTH = 3
_SCATTER_ALPHA = 0.30
_SCATTER_SIZE = 18
_FIG_SIZE = (7, 6)
_DPI = 300

# Optional numeric MALD covariates added to OLS when present in results CSV.
# StressPattern is string/categorical and cannot be z-scored — excluded here.
_OPTIONAL_NUMERIC_COVARIATES = [
    "PhonotacticProbability",
    "TempUP", "OrthUP", "PhonUP",
    "FreqSUBTLEX", "FreqCOCA", "FreqCOCAspok", "FreqGoogle",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_codec_label(codec_name: str, codec_params: str) -> str:
    """Human-readable label for a codec+params combo."""
    try:
        params = json.loads(codec_params)
        parts = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{codec_name} ({parts})" if parts else codec_name
    except Exception:
        return f"{codec_name} ({codec_params})"


def _params_dirname(codec_params: str) -> str:
    """Safe directory name derived from codec params, e.g. 'bandwidth=6.0'."""
    try:
        params = json.loads(codec_params)
        if params:
            return "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return "default"
    except Exception:
        safe = "".join(c if c.isalnum() or c in "=._-" else "_" for c in codec_params)
        return safe[:40] or "default"


def _per_condition_figures_dir(
    codec_name: str, codec_params: str, run_tag: str = ""
) -> pathlib.Path:
    """Unique figure directory for one codec+params+run_tag condition.

    Returns reproduce/figures/codec/{codec_name}/{params_dirname}/{run_tag}/
    e.g.     reproduce/figures/codec/encodec/bandwidth=6.0/TRIM/
             reproduce/figures/codec/encodec/bandwidth=6.0/NONTRIM/
    """
    base = get_codec_figures_dir(codec_name)   # creates …/encodec/
    d = base / _params_dirname(codec_params)
    if run_tag:
        d = d / run_tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_and_validate(csv_path: str) -> pd.DataFrame:
    """Load a results CSV and add derived columns."""
    df = pd.read_csv(csv_path)

    required = ["mel_distance", "n_tokens", "is_word"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {missing}")

    # Lexicality label (string)
    df["lexicality"] = df["is_word"].map(_LEX_LABELS)

    # run_tag: TRIM / NONTRIM (from trim_silence column; default NONTRIM for old CSVs)
    if "trim_silence" in df.columns:
        df["run_tag"] = (
            df["trim_silence"]
            .map({True: "TRIM", False: "NONTRIM", 1: "TRIM", 0: "NONTRIM"})
            .fillna("NONTRIM")
        )
    else:
        df["run_tag"] = "NONTRIM"

    # Codec label (for multi-codec overlays) — includes run_tag
    if "codec_name" in df.columns and "codec_params" in df.columns:
        df["codec_label"] = df.apply(
            lambda r: _make_codec_label(str(r["codec_name"]), str(r["codec_params"]))
                      + f" [{r['run_tag']}]",
            axis=1,
        )
    else:
        df["codec_label"] = csv_path

    # num_phones as integer (may already be int or float)
    if "num_phones" in df.columns:
        df["num_phones_int"] = df["num_phones"].round().astype("Int64")

    return df


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _apply_phone_range(
    df: pd.DataFrame, phones_min: int, phones_max: int
) -> pd.DataFrame:
    """Filter rows to the given num_phones range (inclusive)."""
    if "num_phones_int" not in df.columns:
        return df
    mask = (df["num_phones_int"] >= phones_min) & (df["num_phones_int"] <= phones_max)
    return df[mask]


def _select_usable_covariates(
    df: pd.DataFrame, candidates: list[str], n_min: int = 5
) -> list[str]:
    """Return candidates that have ≥ n_min non-NaN values in BOTH lexicality groups.

    Covariates missing for all pseudowords (e.g. frequency measures) would make
    is_word collinear with the covariate after listwise deletion — they are excluded.
    """
    usable = []
    for col in candidates:
        if col not in df.columns:
            continue
        is_word_mask = df["is_word"].astype(bool)
        n_words = int(df.loc[is_word_mask, col].notna().sum())
        n_pseudo = int(df.loc[~is_word_mask, col].notna().sum())
        if n_words >= n_min and n_pseudo >= n_min:
            usable.append(col)
        else:
            print(
                f"  NOTE: optional covariate '{col}' skipped "
                f"(words={n_words}, pseudo={n_pseudo} valid rows — need ≥{n_min})"
            )
    return usable


# ---------------------------------------------------------------------------
# Per-codec figures
# ---------------------------------------------------------------------------

def _fig1_error_vs_num_phones(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    phones_min: int,
    phones_max: int,
) -> None:
    """Primary: mel distance vs phoneme count, hue=lexicality.

    Scatter points (alpha=0.30) are drawn behind the mean line.
    CI bands are disabled for clarity.
    Phone range always filtered to [phones_min, phones_max].
    """
    if "num_phones_int" not in df.columns:
        print("  SKIP error_vs_num_phones: num_phones column missing")
        return

    plot_df = (
        df.pipe(_apply_phone_range, phones_min, phones_max)
        .dropna(subset=["num_phones_int", "mel_distance"])
        .copy()
    )
    plot_df["num_phones_int"] = plot_df["num_phones_int"].astype(int)

    if plot_df.empty:
        print("  SKIP error_vs_num_phones: no data after phone-range filter")
        return

    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    # Scatter behind the mean line
    for lex, color in _LEX_PALETTE.items():
        sub = plot_df[plot_df["lexicality"] == lex]
        ax.scatter(
            sub["num_phones_int"], sub["mel_distance"],
            color=color, alpha=_SCATTER_ALPHA, s=_SCATTER_SIZE,
            zorder=1, linewidths=0,
        )

    # Mean line (no CI)
    sns.lineplot(
        data=plot_df,
        x="num_phones_int",
        y="mel_distance",
        hue="lexicality",
        marker="o",
        markersize=_MARKER_SIZE,
        linewidth=_LINE_WIDTH,
        palette=_LEX_PALETTE,
        errorbar=None,
        ax=ax,
        zorder=2,
    )

    ax.set_xlabel("Number of Phones", fontsize=24, labelpad=-15)
    ax.set_ylabel("Mel Distance", fontsize=24, labelpad=-15)

    x_vals = sorted(plot_df["num_phones_int"].unique())
    ax.set_xticks(x_vals)
    lo, hi = min(x_vals), max(x_vals)
    ax.set_xticklabels(
        [str(v) if v in (lo, hi) else "" for v in x_vals], fontsize=22
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=16)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_num_phones.png", dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: error_vs_num_phones.png")


def _fig2_error_vs_duration(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    phones_min: int,
    phones_max: int,
    filter_all: bool,
) -> None:
    """Control: mel distance vs binned duration_s, hue=lexicality.

    If filter_all=True, restricts to [phones_min, phones_max] before plotting.
    """
    if "duration_s" not in df.columns:
        print("  SKIP error_vs_duration: duration_s column missing")
        return

    plot_df = df.copy()
    if filter_all:
        plot_df = _apply_phone_range(plot_df, phones_min, phones_max)
    plot_df = plot_df.dropna(subset=["duration_s", "mel_distance"])
    n_bins = max(3, min(6, len(plot_df) // 4))

    if plot_df.empty:
        print("  SKIP error_vs_duration: no data")
        return

    try:
        plot_df["duration_bin"] = pd.qcut(
            plot_df["duration_s"], q=n_bins, duplicates="drop"
        )
        plot_df["duration_mid"] = plot_df["duration_bin"].apply(lambda x: x.mid)
    except ValueError:
        plot_df["duration_mid"] = plot_df["duration_s"].round(2)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    sns.lineplot(
        data=plot_df,
        x="duration_mid",
        y="mel_distance",
        hue="lexicality",
        marker="o",
        markersize=_MARKER_SIZE,
        linewidth=_LINE_WIDTH,
        palette=_LEX_PALETTE,
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("Duration (s)", fontsize=24, labelpad=-15)
    ax.set_ylabel("Mel Distance", fontsize=24, labelpad=-15)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=16)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_duration.png", dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: error_vs_duration.png")


def _fig3_tokens_vs_num_phones(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    phones_min: int,
    phones_max: int,
) -> None:
    """Token count vs phoneme count, hue=lexicality.

    Scatter points (alpha=0.30) drawn behind the mean line; no CI bands.
    Phone range always filtered to [phones_min, phones_max].
    """
    if "num_phones_int" not in df.columns or "n_tokens" not in df.columns:
        print("  SKIP tokens_vs_num_phones: required columns missing")
        return

    plot_df = (
        df.pipe(_apply_phone_range, phones_min, phones_max)
        .dropna(subset=["num_phones_int", "n_tokens"])
        .copy()
    )
    plot_df["num_phones_int"] = plot_df["num_phones_int"].astype(int)

    if plot_df.empty:
        print("  SKIP tokens_vs_num_phones: no data after phone-range filter")
        return

    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    # Scatter behind mean line
    for lex, color in _LEX_PALETTE.items():
        sub = plot_df[plot_df["lexicality"] == lex]
        ax.scatter(
            sub["num_phones_int"], sub["n_tokens"],
            color=color, alpha=_SCATTER_ALPHA, s=_SCATTER_SIZE,
            zorder=1, linewidths=0,
        )

    sns.lineplot(
        data=plot_df,
        x="num_phones_int",
        y="n_tokens",
        hue="lexicality",
        marker="o",
        markersize=_MARKER_SIZE,
        linewidth=_LINE_WIDTH,
        palette=_LEX_PALETTE,
        errorbar=None,
        ax=ax,
        zorder=2,
    )

    ax.set_xlabel("Number of Phones", fontsize=24, labelpad=-15)
    ax.set_ylabel("Token Count", fontsize=24, labelpad=-15)

    x_vals = sorted(plot_df["num_phones_int"].unique())
    ax.set_xticks(x_vals)
    lo, hi = min(x_vals), max(x_vals)
    ax.set_xticklabels(
        [str(v) if v in (lo, hi) else "" for v in x_vals], fontsize=22
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=16)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "tokens_vs_num_phones.png", dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: tokens_vs_num_phones.png")


def _fig4_regression(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    phones_min: int,
    phones_max: int,
    filter_all: bool,
) -> None:
    """OLS: mel_distance ~ num_phones + duration_s + is_word (standardized).

    If filter_all=True, restricts to [phones_min, phones_max] before fitting.
    Saves regression_lexicality.png and regression_summary.txt.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  SKIP regression: statsmodels not installed")
        return

    needed = ["mel_distance", "num_phones", "duration_s", "is_word"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  SKIP regression: missing columns {missing}")
        return

    reg_df = df.copy()
    if filter_all:
        reg_df = _apply_phone_range(reg_df, phones_min, phones_max)

    # Detect optional numeric covariates present and usable in this dataset
    opt_covs = _select_usable_covariates(
        reg_df,
        [c for c in _OPTIONAL_NUMERIC_COVARIATES if c in reg_df.columns],
    )
    reg_df = reg_df[needed + opt_covs].dropna(subset=needed).copy()
    if opt_covs:
        reg_df = reg_df.dropna(subset=opt_covs)

    if len(reg_df) < 10:
        print(f"  SKIP regression: only {len(reg_df)} valid rows (need ≥ 10)")
        return

    # Standardize continuous predictors (base + optional)
    for col in ["num_phones", "duration_s"] + opt_covs:
        mu, sd = reg_df[col].mean(), reg_df[col].std()
        reg_df[f"{col}_z"] = (reg_df[col] - mu) / sd if sd > 0 else 0.0

    reg_df["is_word_num"] = reg_df["is_word"].astype(float)

    opt_terms = " + ".join(f"{c}_z" for c in opt_covs)
    formula = "mel_distance ~ num_phones_z + duration_s_z + is_word_num"
    if opt_terms:
        formula += " + " + opt_terms
    try:
        model = smf.ols(formula, data=reg_df).fit()
    except Exception as exc:
        print(f"  SKIP regression: OLS failed: {exc}")
        return

    # ---- Save text summary ----
    filter_note = f"  (phone range filter: [{phones_min}, {phones_max}])\n" if filter_all else ""
    if opt_covs:
        filter_note += f"  (extra covariates: {', '.join(opt_covs)})\n"
    summary_path = out_dir / "regression_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"OLS Regression: {formula}\n")
        f.write(f"N = {len(reg_df)}\n")
        f.write(filter_note)
        f.write("\n")
        f.write(model.summary().as_text())
    print(f"  Saved: regression_summary.txt  (N={len(reg_df)})")

    # ---- Coefficient bar plot ----
    params = model.params.drop("Intercept", errors="ignore")
    conf = model.conf_int().drop("Intercept", errors="ignore")
    pvals = model.pvalues.drop("Intercept", errors="ignore")

    rename = {
        "num_phones_z": "# Phones (z)",
        "duration_s_z": "Duration (z)",
        "is_word_num": "Is Word",
        **{f"{c}_z": c for c in opt_covs},
    }
    bar_labels = [rename.get(p, p) for p in params.index]
    coefs = params.values
    ci_lo = conf[0].values
    ci_hi = conf[1].values
    errors_lo = coefs - ci_lo
    errors_hi = ci_hi - coefs

    x = np.arange(len(bar_labels))
    colors = ["#e74c3c" if pv < 0.05 else "#95a5a6" for pv in pvals.values]

    fig_w = max(_FIG_SIZE[0], 1.5 * len(bar_labels))
    plt.figure(figsize=(fig_w, _FIG_SIZE[1]))
    plt.bar(x, coefs, color=colors, alpha=0.8, zorder=2)
    plt.errorbar(
        x, coefs,
        yerr=np.array([errors_lo, errors_hi]),
        fmt="none", color="black", capsize=5, linewidth=1.5, zorder=3,
    )
    plt.axhline(0, color="black", linewidth=1)
    _rot = 25 if len(bar_labels) > 4 else 0
    plt.xticks(x, bar_labels, fontsize=13, rotation=_rot,
               ha="right" if _rot else "center")
    plt.ylabel("Standardized Coefficient", fontsize=16)
    plt.title(
        "OLS: mel_distance ~ phones + duration + is_word\n(red = p < 0.05)",
        fontsize=12,
    )
    plt.grid(axis="y", alpha=0.4)

    for xi, pv in zip(x, pvals.values):
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        plt.text(
            xi, max(coefs[xi], 0) + max(errors_hi[xi], 0) + 0.002,
            sig, ha="center", va="bottom", fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(out_dir / "regression_lexicality.png", dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: regression_lexicality.png")


def _run_per_codec(
    df: pd.DataFrame,
    codec_name: str,
    codec_params: str,
    run_tag: str,
    phones_min: int,
    phones_max: int,
    filter_all: bool,
) -> None:
    """Generate all per-codec figures into a condition-unique directory."""
    label = _make_codec_label(codec_name, codec_params) + (f" [{run_tag}]" if run_tag else "")
    out_dir = _per_condition_figures_dir(codec_name, codec_params, run_tag)
    print(f"\n[{label}]  → {out_dir}")

    _fig1_error_vs_num_phones(df, out_dir, phones_min, phones_max)
    _fig2_error_vs_duration(df, out_dir, phones_min, phones_max, filter_all)
    _fig3_tokens_vs_num_phones(df, out_dir, phones_min, phones_max)
    _fig4_regression(df, out_dir, phones_min, phones_max, filter_all)


# ---------------------------------------------------------------------------
# Cross-codec comparison figures
# ---------------------------------------------------------------------------

def _comp_fig1_overlay(
    all_df: pd.DataFrame,
    out_dir: pathlib.Path,
    phones_min: int,
    phones_max: int,
) -> None:
    """Overlay: mel distance vs num_phones; hue=codec_label, style=lexicality."""
    if "num_phones_int" not in all_df.columns:
        print("  SKIP overlay: num_phones column missing")
        return

    plot_df = (
        all_df.pipe(_apply_phone_range, phones_min, phones_max)
        .dropna(subset=["num_phones_int", "mel_distance"])
        .copy()
    )
    plot_df["num_phones_int"] = plot_df["num_phones_int"].astype(int)

    if plot_df.empty:
        print("  SKIP overlay: no data after phone-range filter")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(
        data=plot_df,
        x="num_phones_int",
        y="mel_distance",
        hue="codec_label",
        style="lexicality",
        marker="o",
        markersize=_MARKER_SIZE,
        linewidth=_LINE_WIDTH,
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("Number of Phones", fontsize=20)
    ax.set_ylabel("Mel Distance", fontsize=20)
    ax.legend(fontsize=11, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(
        out_dir / "error_vs_num_phones_overlay.png", dpi=_DPI, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: error_vs_num_phones_overlay.png")


def _comp_fig2_regression_comparison(
    all_df: pd.DataFrame,
    out_dir: pathlib.Path,
    phones_min: int,
    phones_max: int,
    filter_all: bool,
) -> None:
    """Grouped bar: OLS coefficients for each codec side by side.

    If filter_all=True, each group's data is restricted to [phones_min, phones_max].
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  SKIP regression comparison: statsmodels not installed")
        return

    needed = ["mel_distance", "num_phones", "duration_s", "is_word", "codec_label"]
    if any(c not in all_df.columns for c in needed):
        print("  SKIP regression comparison: missing columns")
        return

    # Detect optional numeric covariates usable across the full combined dataset
    opt_covs = _select_usable_covariates(
        all_df,
        [c for c in _OPTIONAL_NUMERIC_COVARIATES if c in all_df.columns],
    )
    opt_terms = " + ".join(f"{c}_z" for c in opt_covs)
    comp_formula = "mel_distance ~ num_phones_z + duration_s_z + is_word_num"
    if opt_terms:
        comp_formula += " + " + opt_terms

    results: list[dict] = []
    rename = {
        "num_phones_z": "# Phones (z)",
        "duration_s_z": "Duration (z)",
        "is_word_num": "Is Word",
        **{f"{c}_z": c for c in opt_covs},
    }

    for label, group in all_df.groupby("codec_label"):
        reg_df = group.copy()
        if filter_all:
            reg_df = _apply_phone_range(reg_df, phones_min, phones_max)
        base_cols = ["mel_distance", "num_phones", "duration_s", "is_word"]
        reg_df = reg_df[base_cols + opt_covs].dropna(subset=base_cols).copy()
        if opt_covs:
            reg_df = reg_df.dropna(subset=opt_covs)
        if len(reg_df) < 10:
            continue
        for col in ["num_phones", "duration_s"] + opt_covs:
            mu, sd = reg_df[col].mean(), reg_df[col].std()
            reg_df[f"{col}_z"] = (reg_df[col] - mu) / sd if sd > 0 else 0.0
        reg_df["is_word_num"] = reg_df["is_word"].astype(float)
        try:
            model = smf.ols(comp_formula, data=reg_df).fit()
        except Exception:
            continue

        params = model.params.drop("Intercept", errors="ignore")
        conf = model.conf_int().drop("Intercept", errors="ignore")
        pvals = model.pvalues.drop("Intercept", errors="ignore")
        for pred in params.index:
            results.append(
                {
                    "codec": label,
                    "predictor": rename.get(pred, pred),
                    "coef": params[pred],
                    "ci_lo": conf.loc[pred, 0],
                    "ci_hi": conf.loc[pred, 1],
                    "pvalue": pvals[pred],
                }
            )

    if not results:
        print("  SKIP regression comparison: no valid models")
        return

    comp_df = pd.DataFrame(results)
    predictors = comp_df["predictor"].unique()
    codecs = comp_df["codec"].unique()
    x = np.arange(len(predictors))
    width = 0.8 / len(codecs)
    palette = sns.color_palette("colorblind", len(codecs))

    plt.figure(figsize=(max(7, 2 * len(predictors)), 6))
    for i, (codec_label, color) in enumerate(zip(codecs, palette)):
        sub = comp_df[comp_df["codec"] == codec_label].set_index("predictor")
        offsets = x + (i - len(codecs) / 2 + 0.5) * width
        coefs = [sub.loc[p, "coef"] if p in sub.index else 0.0 for p in predictors]
        ci_lo = [sub.loc[p, "ci_lo"] if p in sub.index else 0.0 for p in predictors]
        ci_hi = [sub.loc[p, "ci_hi"] if p in sub.index else 0.0 for p in predictors]
        err_lo = [c - lo for c, lo in zip(coefs, ci_lo)]
        err_hi = [hi - c for c, hi in zip(coefs, ci_hi)]
        plt.bar(offsets, coefs, width, color=color, alpha=0.8, label=codec_label, zorder=2)
        plt.errorbar(
            offsets, coefs,
            yerr=[err_lo, err_hi],
            fmt="none", color="black", capsize=3, linewidth=1, zorder=3,
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, predictors, fontsize=13)
    plt.ylabel("Standardized Coefficient", fontsize=15)
    plt.title(
        "OLS Coefficient Comparison (mel_distance ~ phones + duration + is_word)",
        fontsize=11,
    )
    plt.legend(fontsize=11)
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "regression_comparison.png", dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: regression_comparison.png")


# ---------------------------------------------------------------------------
# Summary tables (Task B)
# ---------------------------------------------------------------------------

def _save_summary_tables(
    all_df: pd.DataFrame,
    out_dir: pathlib.Path,
    phones_min: int,
    phones_max: int,
) -> None:
    """Compute and save summary_table.csv and length_effect_slopes.csv.

    summary_table.csv — per (codec_label, lexicality, num_phones_int) means:
      n, mean_mel_distance, mean_si_sdr, mean_n_tokens

    length_effect_slopes.csv — OLS slope of mean_mel_distance on num_phones
    per (codec_label, lexicality), fit on the aggregated row means:
      n_points, slope, intercept, r_squared
    """
    if "num_phones_int" not in all_df.columns or "mel_distance" not in all_df.columns:
        print("  SKIP summary tables: num_phones_int or mel_distance missing")
        return

    # Filter to phone range
    filt_df = _apply_phone_range(all_df, phones_min, phones_max).copy()
    filt_df["num_phones_int"] = filt_df["num_phones_int"].astype(int)

    try:
        group_cols = ["codec_label", "lexicality", "num_phones_int"]
        agg_cols = {
            "mel_distance": ["count", "mean"],
            **{col: "mean" for col in ["si_sdr", "n_tokens"] if col in filt_df.columns},
        }

        agg = filt_df.groupby(group_cols).agg(agg_cols)
        agg.columns = [
            "_".join(filter(None, c)) if isinstance(c, tuple) else c
            for c in agg.columns
        ]
        agg = agg.rename(columns={"mel_distance_count": "n", "mel_distance_mean": "mean_mel_distance"})
        if "si_sdr_mean" in agg.columns:
            agg = agg.rename(columns={"si_sdr_mean": "mean_si_sdr"})
        if "n_tokens_mean" in agg.columns:
            agg = agg.rename(columns={"n_tokens_mean": "mean_n_tokens"})
        agg = agg.reset_index()

        summary_path = out_dir / "summary_table.csv"
        agg.to_csv(summary_path, index=False)
        print(f"  Saved: summary_table.csv  ({len(agg)} rows)")

        # ---- Length effect slopes ----
        slope_rows: list[dict] = []
        for (codec_label, lexicality), grp in agg.groupby(["codec_label", "lexicality"]):
            sub = grp.dropna(subset=["num_phones_int", "mean_mel_distance"]).sort_values("num_phones_int")
            x = sub["num_phones_int"].values.astype(float)
            y = sub["mean_mel_distance"].values.astype(float)
            n_pts = len(x)

            if n_pts < 2:
                slope_rows.append(
                    dict(codec_label=codec_label, lexicality=lexicality,
                         n_points=n_pts, slope=np.nan, intercept=np.nan, r_squared=np.nan)
                )
                continue

            coeffs = np.polyfit(x, y, 1)   # [slope, intercept]
            slope, intercept = float(coeffs[0]), float(coeffs[1])

            # R² from Pearson correlation
            r = float(np.corrcoef(x, y)[0, 1])
            r_squared = r ** 2 if not np.isnan(r) else np.nan

            slope_rows.append(
                dict(codec_label=codec_label, lexicality=lexicality,
                     n_points=n_pts, slope=slope, intercept=intercept, r_squared=r_squared)
            )

        slopes_df = pd.DataFrame(slope_rows)
        slopes_path = out_dir / "length_effect_slopes.csv"
        slopes_df.to_csv(slopes_path, index=False)
        print(f"  Saved: length_effect_slopes.csv  ({len(slopes_df)} rows)")

        # Print a compact view for quick inspection
        print()
        print("  Length effect slopes (mel_distance ~ num_phones, on aggregated means):")
        print(f"  {'codec':<30} {'lex':<8} {'slope':>8} {'R²':>6}")
        for _, row in slopes_df.iterrows():
            slope_str = f"{row['slope']:.5f}" if not np.isnan(row['slope']) else "   nan"
            r2_str = f"{row['r_squared']:.3f}" if not np.isnan(row['r_squared']) else "  nan"
            print(f"  {str(row['codec_label']):<30} {str(row['lexicality']):<8} {slope_str:>8} {r2_str:>6}")

    except Exception as exc:
        print(f"  WARNING: summary tables failed ({exc}); skipping. Figures will still be generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results", nargs="+", required=True,
        metavar="CSV",
        help="One or more results CSVs from scripts/codec_reconstruct.py",
    )
    p.add_argument(
        "--phones-min", type=int, default=3, dest="phones_min",
        help="Minimum num_phones for phone-indexed plots and summary tables (default: 3)",
    )
    p.add_argument(
        "--phones-max", type=int, default=9, dest="phones_max",
        help="Maximum num_phones for phone-indexed plots and summary tables (default: 9)",
    )
    p.add_argument(
        "--filter-all", action="store_true", dest="filter_all",
        help=(
            "Apply phone range filter globally: also restricts error_vs_duration, "
            "OLS regression, and regression_comparison to [phones-min, phones-max]. "
            "Default: only phone-indexed plots are filtered."
        ),
    )
    p.add_argument(
        "--out-suffix", default="", dest="out_suffix",
        metavar="SUFFIX",
        help=(
            "Suffix appended to 'comparison' for the comparison output directory "
            "(e.g. TRIM → comparison__TRIM, NONTRIM → comparison__NONTRIM, "
            "MIX → comparison__MIX). Default: empty → comparison/."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load and validate all CSVs
    frames: list[pd.DataFrame] = []
    for csv_path in args.results:
        print(f"Loading {csv_path} …")
        try:
            df = _load_and_validate(csv_path)
        except Exception as exc:
            print(f"  ERROR loading {csv_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        frames.append(df)
        print(f"  {len(df)} rows  |  columns: {list(df.columns)}")

    all_df = pd.concat(frames, ignore_index=True)

    filter_note = " (--filter-all: applied globally)" if args.filter_all else ""
    print(f"\nPhone range: [{args.phones_min}, {args.phones_max}]{filter_note}")

    # Group by codec_name + codec_params + run_tag (run_tag added by _load_and_validate)
    if "codec_name" not in all_df.columns:
        all_df["codec_name"] = "unknown"
    if "codec_params" not in all_df.columns:
        all_df["codec_params"] = "{}"
    group_cols = ["codec_name", "codec_params", "run_tag"]

    groups = all_df.groupby(group_cols)
    print(f"\nFound {len(groups)} codec group(s):")
    for (cn, cp, rt), grp in groups:
        label = _make_codec_label(str(cn), str(cp)) + (f" [{rt}]" if rt else "")
        cond_dir = _per_condition_figures_dir(str(cn), str(cp), str(rt))
        print(f"  {label}  ({len(grp)} rows)  → {cond_dir}")

    # Comparison directory (respects --out-suffix)
    comp_dir = _ensure_dir(get_codec_comparison_figures_dir(args.out_suffix))

    # Summary tables (always, regardless of number of CSVs)
    print(f"\n[Summary tables]  → {comp_dir}")
    _save_summary_tables(all_df, comp_dir, args.phones_min, args.phones_max)

    # Per-codec figures
    for (codec_name, codec_params, run_tag), grp in groups:
        _run_per_codec(
            grp, str(codec_name), str(codec_params), str(run_tag),
            args.phones_min, args.phones_max, args.filter_all,
        )

    # Cross-codec comparison figures (only when 2+ distinct groups)
    if len(groups) >= 2:
        print(f"\n[Cross-codec comparison]  → {comp_dir}")
        _comp_fig1_overlay(all_df, comp_dir, args.phones_min, args.phones_max)
        _comp_fig2_regression_comparison(
            all_df, comp_dir, args.phones_min, args.phones_max, args.filter_all,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
