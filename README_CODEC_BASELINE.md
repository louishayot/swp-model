# Codec Reconstruction Baseline — Track A (MALD / EnCodec)

**Goal**: test whether a pretrained neural audio codec (EnCodec) reproduces the
lexicality and length effects seen in human nonword repetition and the SWP LSTM,
using round-trip reconstruction error on naturally-recorded speech as the proxy.

Outputs (CSVs, figures) are **gitignored** and produced locally by running the
commands below. Nothing in `reproduce/data/`, `reproduce/figures/`, or
`data/external/` is tracked.

---

## Quickstart

### Prerequisites

```bash
pip install -r requirements.txt
```

Key codec dependencies (already in `requirements.txt`):
- `transformers>=4.36.0` — EnCodec model (HuggingFace)
- `torchaudio>=2.5.0` — audio loading and resampling
- `torchcodec>=0.1` — required by `torchaudio.load()`; without it you get an `ImportError` at runtime even though `import torchaudio` succeeds
- `statsmodels>=0.14.4` — OLS regression in the analysis script

---

### Step 0 — sanity check (no audio needed)

Verify EnCodec loads and round-trips a synthetic waveform:

```bash
python scripts/sanity_encodec.py
python scripts/sanity_encodec.py --bandwidth 1.5
python scripts/sanity_encodec.py --sr 44100 --duration 1.0
```

What to look for:
- Script exits `[PASS]` with all metrics finite — absolute values depend on signal type and device; white noise can yield low or negative SI-SDR, that is expected.
- `reencode_consistency` should be > 0 (token codes are assigned and stable).

---

### Step 1 — obtain MALD audio (manual, one-time)

MALD (Massive Auditory Lexical Decision) is a publicly available dataset of
professionally-recorded English words and pseudowords.

1. Download from [ERA / University of Alberta](https://nascl.rc.nau.edu/resources/massive-auditory-lexical-decision/):
   - `MALD1_1_ItemData.txt` — tab-delimited metadata
   - Word audio: DOI [10.7939/r3-v0jr-rr12](https://doi.org/10.7939/r3-v0jr-rr12)
   - Pseudoword audio: DOI [10.7939/r3-v7jh-p314](https://doi.org/10.7939/r3-v7jh-p314)
2. Place files under (gitignored):
   ```
   data/external/mald/
   ├── MALD1_1_ItemData.txt
   ├── words/          ← extracted word WAVs
   └── pseudowords/    ← extracted pseudoword WAVs
   ```

---

### Step 2 — build the balanced subset

```bash
# Smoke-test subset (20 items: 10 words + 10 pseudowords)
python scripts/setup_mald.py --tiny
# → data/external/mald/subset_tiny.csv

# Full subset (400 items: 200 words + 200 pseudowords)
python scripts/setup_mald.py
# → data/external/mald/subset.csv
# → data/external/mald/subset_diagnostics.txt   (balance report)
```

The script:
- Restricts sampling to `num_phones` in [`--phones-min`, `--phones-max`] (default 3–9), stratified by phone count within each lexicality group (seed=42 for reproducibility). Matches the CCN paper range out of the box.
- Includes optional MALD columns when present: `PhonotacticProbability`, `StressPattern`, `TempUP`, `OrthUP`, `PhonUP`, `FreqSUBTLEX`, `FreqCOCA`, `FreqCOCAspok`, `FreqGoogle`
- Prints column mapping, speaker scan, and phone/duration distribution diagnostics.

---

### Step 3 — run codec reconstructions

**Non-trimmed (raw waveform; recommended for first run):**

```bash
python scripts/codec_reconstruct.py \
    --codec encodec --codec-arg bandwidth=6.0 \
    --dataset data/external/mald/subset.csv \
    --output reproduce/data/codec/encodec/subset__bw6_NONTRIM.csv

python scripts/codec_reconstruct.py \
    --codec encodec --codec-arg bandwidth=1.5 \
    --dataset data/external/mald/subset.csv \
    --output reproduce/data/codec/encodec/subset__bw1p5_NONTRIM.csv
```

**Silence-trimmed (leading/trailing silence removed before encoding):**

> Always set `--output` explicitly for trimmed runs — the auto-generated filename
> uses a hash of codec kwargs only and would collide with the non-trimmed run at
> the same bandwidth.

```bash
python scripts/codec_reconstruct.py \
    --codec encodec --codec-arg bandwidth=6.0 \
    --dataset data/external/mald/subset.csv \
    --trim-silence \
    --output reproduce/data/codec/encodec/subset__bw6_TRIM.csv

python scripts/codec_reconstruct.py \
    --codec encodec --codec-arg bandwidth=1.5 \
    --dataset data/external/mald/subset.csv \
    --trim-silence \
    --output reproduce/data/codec/encodec/subset__bw1p5_TRIM.csv
```

Each run produces:
- `<output>.csv` — 400-row results table (metrics + metadata)
- `<output>.meta.json` — full parameter record for reproducibility

Use `--regenerate` to overwrite an existing cached CSV.

---

### Step 4 — run analysis

```bash
# Single condition
python reproduce/scripts/codec_analysis.py \
    --results reproduce/data/codec/encodec/subset__bw6_NONTRIM.csv \
    --phones-min 3 --phones-max 9 --filter-all

# Bandwidth comparison (NONTRIM)
python reproduce/scripts/codec_analysis.py \
    --results reproduce/data/codec/encodec/subset__bw6_NONTRIM.csv \
              reproduce/data/codec/encodec/subset__bw1p5_NONTRIM.csv \
    --phones-min 3 --phones-max 9 --filter-all \
    --out-suffix NONTRIM

# Bandwidth comparison (TRIM)
python reproduce/scripts/codec_analysis.py \
    --results reproduce/data/codec/encodec/subset__bw6_TRIM.csv \
              reproduce/data/codec/encodec/subset__bw1p5_TRIM.csv \
    --phones-min 3 --phones-max 9 --filter-all \
    --out-suffix TRIM

# All four runs together (mixed; useful for TRIM vs NONTRIM overlay)
python reproduce/scripts/codec_analysis.py \
    --results reproduce/data/codec/encodec/subset__bw6_NONTRIM.csv \
              reproduce/data/codec/encodec/subset__bw1p5_NONTRIM.csv \
              reproduce/data/codec/encodec/subset__bw6_TRIM.csv \
              reproduce/data/codec/encodec/subset__bw1p5_TRIM.csv \
    --phones-min 3 --phones-max 9 --filter-all \
    --out-suffix MIX
```

---

## Outputs

All outputs go under `reproduce/` (gitignored):

```
reproduce/data/codec/encodec/
├── subset__bw6_NONTRIM.csv          ← 400 rows
├── subset__bw6_NONTRIM.meta.json    ← full params
├── subset__bw1p5_NONTRIM.csv
...

reproduce/figures/codec/
├── encodec/
│   ├── bandwidth=6.0/
│   │   ├── NONTRIM/
│   │   │   ├── error_vs_num_phones.png      ← primary: mel distance vs phoneme count
│   │   │   ├── error_vs_duration.png        ← control: mel distance vs duration (binned)
│   │   │   ├── tokens_vs_num_phones.png     ← codec token count vs phoneme count
│   │   │   ├── regression_lexicality.png    ← OLS coefficient bar chart
│   │   │   └── regression_summary.txt       ← full statsmodels OLS table
│   │   └── TRIM/  (same 5 files)
│   └── bandwidth=1.5/
│       └── NONTRIM/ and TRIM/
├── comparison__NONTRIM/
│   ├── error_vs_num_phones_overlay.png      ← bw=6.0 vs bw=1.5 overlay
│   ├── regression_comparison.png            ← grouped OLS coefficient bar chart
│   ├── summary_table.csv
│   └── length_effect_slopes.csv
├── comparison__TRIM/   (same)
└── comparison__MIX/    (same)
```

**How figure collisions are avoided**: each CSV row carries a `trim_silence`
boolean. The analysis derives `run_tag = "TRIM" / "NONTRIM"` and routes per-condition
figures to a subdirectory of that name. `codec_label` in every figure includes the tag
(e.g. `"encodec (bandwidth=6.0) [NONTRIM]"`). Comparison directories are further
disambiguated by `--out-suffix`.

**What to look for in the figures:**
- `error_vs_num_phones.png`: does mel distance increase with phoneme count? Is the
  slope steeper for pseudowords than words (lexicality × length interaction)?
- `regression_lexicality.png`: is the `Is Word` bar significantly different from zero
  (red = p < 0.05) after controlling for duration and phoneme count?
- `regression_summary.txt`: check N — listwise deletion from frequency covariates
  (NaN for pseudowords) can reduce sample size; verify both groups remain represented.

---

## CSV schema

| Column | Description |
|---|---|
| `word`, `is_word` | Item identity and lexicality |
| `num_phones`, `num_sylls` | From MALD metadata |
| `duration_s` | Measured from waveform (post-trim if `--trim-silence`) |
| `duration_meta_s` | MALD metadata duration (cross-check only) |
| `duration_pretrim_s` | Pre-trim duration (only with `--trim-silence`) |
| `silence_ratio` | Fraction of frames below RMS threshold (always written) |
| `codec_name` | e.g. `"encodec"` |
| `codec_params` | JSON, e.g. `'{"bandwidth": 6.0}'` |
| `trim_silence` | Boolean — routes figures to TRIM/NONTRIM subdir |
| `n_tokens` | Total discrete tokens (Q × T frames) |
| `si_sdr` | Scale-Invariant SDR in dB (higher = better) |
| `mel_distance` | Log-mel L1 distance (lower = better) |
| `reencode_consistency` | Re-encode token agreement ∈ [0, 1] |
| `PhonotacticProbability`, `FreqSUBTLEX`, … | Optional MALD columns if present |

---

## Adding a new codec (DAC, Mimi, …)

Three steps, no changes to the pipeline:

**1. Create `swp/codecs/dac_codec.py`** implementing the `AudioCodec` Protocol:

```python
from swp.codecs.base import AudioCodec, ReconstructionResult
import torch

class DACCodec:
    def __init__(self, model_type: str = "44khz", device=None): ...

    @property
    def sample_rate(self) -> int: return 44100   # or 16000/24000

    @property
    def name(self) -> str: return "dac"

    def reconstruct(self, wav: torch.Tensor, sr: int) -> ReconstructionResult:
        # resample if sr != self.sample_rate, encode, decode, return dict
        ...
```

**2. Register in `swp/codecs/registry.py`** (one line):

```python
_CODEC_REGISTRY["dac"] = ("swp.codecs.dac_codec", "DACCodec")
```

**3. Run with the same CLI**:

```bash
python scripts/codec_reconstruct.py \
    --codec dac --codec-arg model_type=44khz \
    --dataset data/external/mald/subset.csv \
    --output reproduce/data/codec/dac/subset__dac_44khz.csv

python reproduce/scripts/codec_analysis.py \
    --results reproduce/data/codec/encodec/subset__bw6_NONTRIM.csv \
              reproduce/data/codec/dac/subset__dac_44khz.csv \
    --phones-min 3 --phones-max 9 --filter-all \
    --out-suffix COMPARE
```

No changes to metrics, pipeline logic, CSV schema, or analysis code.

---

## New files on this branch

| File | Purpose |
|---|---|
| `swp/codecs/__init__.py` | Package init |
| `swp/codecs/base.py` | `AudioCodec` Protocol + `ReconstructionResult` TypedDict |
| `swp/codecs/registry.py` | Lazy-loading codec registry |
| `swp/codecs/encodec_codec.py` | HF `facebook/encodec_24khz` wrapper |
| `swp/codecs/metrics.py` | `si_sdr`, `mel_distance`, `reencode_consistency`, `compute_all` |
| `scripts/sanity_encodec.py` | In-memory smoke test (no audio files) |
| `scripts/setup_mald.py` | MALD metadata parser → balanced subset CSV |
| `scripts/codec_reconstruct.py` | Codec-agnostic reconstruction pipeline |
| `reproduce/scripts/codec_analysis.py` | Figures + OLS regression (multi-codec capable) |

**Modified**: `swp/utils/paths.py` (+4 path helpers), `.gitignore` (`data/external/`, `stimuli/audio/`).
