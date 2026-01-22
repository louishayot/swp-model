# A Neural Model for Word Repetition

> Abstract
>
> Word repetition — hearing a word and repeating it aloud — is a skill that takes years to develop in children, poses challenges for adults learning new languages, and can break down after brain damage. Cognitive science proposes a multi-component model for this task, but the underlying neural mechanisms remain unclear. To bridge this gap, we train deep neural networks on word repetition and probe them with tests inspired by human behavioral studies. We also simulate brain damage through ablation studies, creating "patient models" whose errors can be compared to clinical speech errors. Our results show that neural models can reproduce several human-like effects, while also diverging in important ways, pointing to both the promise and the challenges of developing biologically grounded models of language.

Neural models for single-word processing with an auditory repetition pathway. This repo supports training from scratch, evaluation on controlled datasets, and reproducing the paper's figures and analyses. (swp = single word processing)

[![arXiv](https://img.shields.io/badge/arXiv-2506.13450-b31b1b.svg)](https://arxiv.org/abs/2506.13450)

## Table of contents

- [Setup](#setup)
- [Training](#training)
- [Acoustic pathway (Ua_w2v)](#acoustic-pathway-ua_w2v)
- [Load weights](#load-weights)
- [Repository structure](#repository-structure)
- [Reproduce the paper figures](#reproduce-the-paper-figures)
- [Reproducibility practices](#reproducibility-practices)
- [Troubleshooting](#troubleshooting)
- [Experimental: Sentence-level acoustic extraction](#experimental-sentence-level-acoustic-extraction)
- [Citations](#citations)

## Setup

```bash
git clone git@github.com:danieldager/swp-model.git
cd swp-model
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_lg
pip install git+https://github.com/LouisJalouzot/MLEM_minimal.git
```

Optional conda

```bash
conda create -n swpm python=3.11 -y
conda activate swpm
```

Optional pyenv

```bash
pyenv install 3.11.0
pyenv virtualenv 3.11.0 swpm
pyenv activate swpm
```

## Training

Run the training script with your hyperparameters:

```bash
python scripts/train_repetition.py \
	--num_epochs 75 \
	--batch_size 1024 \
	--recur_type lstm \
	--hidden_size 128 \
	--num_layers 1 \
	--learn_rate 0.001 \
	--dropout 0.0 \
	--tf_ratio 0.0 \
	--seed 42 \
	--verbose
```

What gets saved where

- Checkpoints after each epoch (and 10 checkpoints in the first epoch): `weights/<model_name>/<train_name>/<epoch>.pth`
- Names are auto-generated for traceability:
  - `model_name` example: `Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1`
  - `train_name` example: `b1024_l0.001_fall_s42_sn_ec`

## Acoustic pathway (Ua_w2v)

> **Branch**: `feature/acoustic-wav2vec`

The acoustic pathway (`Ua_w2v`) replaces phoneme-input with frozen wav2vec2 features while keeping the exact same phoneme decoder as `Ua`. This enables direct comparison between oracle phoneme input and realistic acoustic input on single-word repetition.

**Key points:**
- Uses Speech Commands dataset (35 words, isolated utterances) for word-level baseline
- Same decoder architecture, vocabulary (42 phonemes), and training procedure as `Ua`
- For sentence-level experiments (not comparable to Ua), see [Experimental: Sentence-level acoustic extraction](#experimental-sentence-level-acoustic-extraction)

### Prerequisites

```bash
pip install transformers datasets torchaudio
```

### Quick start

Run the full end-to-end smoke test:

```bash
./run_acoustic_smoke.sh
```

Verify auto-detection picks correct models:

```bash
python scripts/test_acoustic.py --manifest_path ./acoustic_words_overfit/manifest.json --verbose
python scripts/test_acoustic.py --manifest_path ./acoustic_words_train/manifest.json --verbose
```

Generate learning curve figures:

```bash
python scripts/make_acoustic_figures.py --output_dir ./docs/figs/ua_w2v
```

### Outputs

**Logs and weights:**
- Training logs: `./logs_acoustic_words/`
- Model checkpoints: `./weights/Ua_w2v_LSTM_h128_l1_v42_d0.0_t0.0_s1__gi768/<train_name>/<epoch>.pth`
- Run metadata: `./weights/.../run_info.json` (contains `manifest_path` for auto-detection)

**Auto-detection:**
Test scripts auto-detect the correct trained model by filtering `run_info.json` files that match the provided `manifest_path`. If multiple matches exist, the most recent is selected.

**Figures:**
- Learning curves: `./docs/figs/ua_w2v/learning_curve_*.png`

### Code changes in this branch

**Added:**
- `scripts/train_acoustic.py` — Training script for Ua_w2v
- `scripts/test_acoustic.py` — Evaluation with manifest-based auto-detect
- `scripts/make_acoustic_figures.py` — Generate learning curve PNGs from logs
- `swp/models/acoustic_encoder.py` — AcousticEncoder (projection + LSTM + pack_padded_sequence)
- `swp/datasets/acoustic.py` — Acoustic dataloaders
- `swp/train/acoustic.py` — Training loop (saves run_info.json)
- `run_acoustic_smoke.sh` — End-to-end smoke test (b10 overfit + b32 full)
- `docs/figs/ua_w2v/*.png` — Curated learning curves

**Modified:**
- `swp/models/autoencoder.py` — Added AcousticUnimodel
- `swp/utils/models.py` — Added Ua_w2v parsing + instantiation
- `swp/utils/paths.py` — Path helpers for acoustic artifacts
- `README.md`, `.gitignore` — Documentation + ignore rules

**Behavioral fixes:**
- `trust_remote_code=True` for Speech Commands loading (no interactive prompts)
- Test auto-detect filters by normalized `manifest_path` match (picks most recent if multiple)

### Example figures

**Overfit sanity check (b10, 10 samples of "yes"):**

![Overfit learning curves](docs/figs/ua_w2v/learning_curve_overfit_b10.png)

Reaches 0 errors within 50 epochs, validating the acoustic encoder → phoneme decoder architecture.

**Full word-level training (b32, 3500 samples, 35 words × 100 utterances):**

![Full training learning curves](docs/figs/ua_w2v/learning_curve_full_b32.png)

Achieves 100% accuracy at epoch 50, demonstrating successful word-level repetition from frozen wav2vec2 features.

### Architecture

**Model naming** (follows Ua convention):
- `model_name`: `Ua_w2v_LSTM_h128_l1_v42_d0.0_t0.0_s1__gi768`
  - `Ua_w2v` = auditory pathway with wav2vec2
  - `gi768` = input dimension (768 for wav2vec2-base)
- `train_name`: `b32_l0.001_fall_s42_sn_ec`

**Design:**
- Frozen wav2vec2-base (offline feature extraction, no fine-tuning)
- Encoder: Linear projection (768→128) + LSTM (h=128, l=1) + `pack_padded_sequence`
- Decoder: Identical to Ua (DecoderLSTM, vocab=42, teacher forcing, cross-entropy)

### Load trained models

```python
from swp.utils.models import get_model, load_weights
from swp.utils.setup import set_device

model_name = "Ua_w2v_LSTM_h128_l1_v42_d0.0_t0.0_s1__gi768"
train_name = "b32_l0.001_fall_s42_sn_ec"
checkpoint = "50"

device = set_device()
model = get_model(model_name)
load_weights(model, model_name, train_name, checkpoint, device)
```

### Next steps: Paper-aligned TTS wordlists (WFE/SSP)

Speech Commands is a sanity baseline but not aligned with the paper's curated wordlists. To enable paper-aligned behavioral analyses (WFE, SSP, unit-49 ablation, early-EOS signatures):

- [ ] Export SWP train/eval wordlists to .txt
- [ ] Generate TTS audio (gTTS, pyttsx3, or Tacotron2)
- [ ] Extract wav2vec2 features from TTS audio
- [ ] Train Ua_w2v on SWP wordlists
- [ ] Run paper-aligned eval scripts (`reproduce/scripts/`)
- [ ] Generate WFE/SSP figures, unit ablations, length effects

Only with paper-aligned wordlists can we test whether acoustic models exhibit the same behavioral signatures as phoneme-input models.

## Load weights

Load trained `Ua` (phoneme-input) models:

```python
from swp.utils.models import get_model, load_weights
from swp.utils.setup import set_device

model_name = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
train_name = "b1024_l0.001_fall_s42_sn_ec"
checkpoint = "75"

device = set_device()
model = get_model(model_name)
load_weights(model=model, model_name=model_name, train_name=train_name, checkpoint=checkpoint, device=device)
```

Alternatively, evaluate using the test script:

```bash
python scripts/test_repetition.py \
	--model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 \
	--train_name b1024_l0.001_fall_s42_sn_ec \
	--checkpoint 75 \
	--batch_size 1024 \
	--verbose
```

**Outputs:**
- Results: `results/evaluation/<model_name>/<train_name>/<checkpoint>/`
- Figures: `results/figures/<model_name>/<train_name>/<checkpoint>/evaluation/`

## Repository structure

```
swp-model/
├── swp/                    # Core package
│   ├── datasets/           # Phoneme folds, acoustic dataloaders
│   ├── models/             # Encoders, decoders (AcousticEncoder, DecoderLSTM)
│   ├── train/              # Training loops (repetition.py, acoustic.py)
│   ├── test/               # Evaluation logic
│   ├── viz/                # Plotting utilities
│   └── utils/              # Paths, model parsing, seeding
├── scripts/                # CLI entry points
│   ├── train_repetition.py, train_acoustic.py
│   ├── test_repetition.py, test_acoustic.py
│   ├── make_acoustic_figures.py
│   └── train_repetition.sh (Slurm wrapper)
├── reproduce/              # Paper figure reproduction
│   ├── scripts/            # Behavioral, embeddings, ablations
│   └── reproduce.ipynb
├── stimuli/                # Folds, morphemes, handmade stimuli
├── weights/                # Model checkpoints (auto-created)
├── results/                # Evaluation outputs (auto-created)
└── docs/figs/              # Curated figures for README
```

**Key conventions:**
- `model_name`: architecture + hyperparameters (LSTM_h128_l1_v42_d0.0_t0.0_s1)
- `train_name`: training regime (b1024_l0.001_fall_s42_sn_ec)

## Reproduce the paper figures

Run all analyses:

```bash
cd reproduce/scripts
python run_all.py
```

Run individual analysis:

```bash
cd reproduce/scripts
python univariate_analysis.py \
	--model-name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 \
	--weights-path weights/1024_75.pth \
	--batch-size 1024 \
	--hidden-size 128
```

Use your own trained weights by passing `weights/<model_name>/<train_name>/<epoch>.pth` to `--weights-path`.

## Reproducibility practices

- **Seeding**: `seed_everything(42)` called consistently (`swp/utils/setup.py`)
- **Paths**: Centralized in `swp/utils/paths.py` (auto-switches to `$WORK` on Jean Zay)
- **Caching**: Scripts save CSVs/NPYs to avoid recomputation; use `--regenerate` to refresh
- **Naming**: `model_name` and `train_name` encode full configuration for transparency
- **Environment**: Export `pip freeze > requirements.lock` for archival

## Troubleshooting

- **FileNotFoundError for weights**: Run from repo root or use absolute paths. Format: `weights/<model_name>/<train_name>/<epoch>.pth`
- **Missing packages**: Install with `pip install -r requirements.txt`
- **CUDA not found**: Check `python -c "import torch; print(torch.cuda.is_available())"`

## Experimental: Sentence-level acoustic extraction

> ⚠️ **WARNING**: Sentence-level LibriSpeech extraction is **NOT comparable** to single-word Ua models. Use Speech Commands (word-level) for paper-aligned experiments.

LibriSpeech extraction differs significantly from single-word repetition:
- Variable length (50-100+ phonemes vs. 3-8 for words)
- Much harder task (early models produce degenerate outputs)
- Not aligned with paper evaluations (WFE, SSP, ablations assume single words)

**Usage** (experimental only):

```bash
python scripts/extract_features.py \
    --output_dir ./acoustic_features_sentences \
    --split validation.clean \
    --limit 100
```

For paper-aligned work, use Speech Commands (word-level) or the recommended TTS approach (see "Next steps" above).

## Citations

Hannagan, T., Agrawal, A., Cohen, L., & Dehaene, A. S. (2021). Emergence of a compositional neural code for written words: Recycling of a convolutional neural network for reading. Proceedings of the National Academy of Sciences, 118(46), e2104779118.

Agrawal, A., & Dehaene, S. (2024). Cracking the neural code for word recognition in convolutional neural networks. arXiv preprint arXiv:2403.06159.

Agrawal, A., & Dehaene, S. (2023). Dissecting the neuronal mechanisms of invariant word recognition. bioRxiv, 2023-11.

Kubilius, J., Schrimpf, M., Nayebi, A., Bear, D., Yamins, D.L.K., DiCarlo, J.J. (2018) CORnet: Modeling the Neural Mechanisms of Core Object Recognition. biorxiv. doi:10.1101/408385

Kubilius, J., Schrimpf, M., Kar, K., Rajalingham, R., Hong, H., Majaj, N., ... & Dicarlo, J. (2019). Brain-like object recognition with high-performing shallow recurrent ANNs. In Advances in Neural Information Processing Systems (pp. 12785-12796).

Burgess, N., & Hitch, G. J. (1992). Toward a network model of the articulatory loop. Journal of memory and language, 31(4), 429-460.

Botvinick, M. M., & Plaut, D. C. (2006). Short-term memory for serial order: a recurrent neural network model. Psychological review, 113(2), 201.

Sajid, N., Holmes, E., Costa, L. D., Price, C., & Friston, K. (2022). A mixed generative model of auditory word repetition. bioRxiv, 2022-01.