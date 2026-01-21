# A Neural Model for Word Repetition

> Abstract
>
> Word repetition — hearing a word and repeating it aloud — is a skill that takes years to develop in children, poses challenges for adults learning new languages, and can break down after brain damage. Cognitive science proposes a multi-component model for this task, but the underlying neural mechanisms remain unclear. To bridge this gap, we train deep neural networks on word repetition and probe them with tests inspired by human behavioral studies. We also simulate brain damage through ablation studies, creating “patient models” whose errors can be compared to clinical speech errors. Our results show that neural models can reproduce several human-like effects, while also diverging in important ways, pointing to both the promise and the challenges of developing biologically grounded models of language.

Neural models for single-word processing with an auditory repetition pathway. This repo supports training from scratch, evaluation on controlled datasets, and reproducing the paper’s figures and analyses. (swp = single word processing)

[![arXiv](https://img.shields.io/badge/arXiv-2506.13450-b31b1b.svg)](https://arxiv.org/abs/2506.13450)

## Table of contents

- Setup
- Training
- Acoustic pathway (Ua_w2v)
- Load weights
- Repository structure
- Reproduce the paper figures
- Reproducibility practices (seeds, paths, caching)
- Troubleshooting
- Experimental: Sentence-level acoustic extraction
- Citations

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

### Training

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

The acoustic pathway (`Ua_w2v`) uses frozen wav2vec2 features as input instead of phoneme sequences, while keeping the **same phoneme decoder** as `Ua`. This enables direct comparison between phoneme-input and acoustic-input models on the single-word repetition task.

**Why word-level audio?** The paper studies single-word repetition, not sentence-level speech recognition. We use the [Speech Commands](https://huggingface.co/datasets/google/speech_commands) dataset (isolated single-word utterances) to match this setting. For sentence-level experiments (not comparable to Ua), see [Experimental: Sentence-level acoustic extraction](#experimental-sentence-level-acoustic-extraction).

### Prerequisites

```bash
pip install transformers datasets torchaudio
```

Note: The Speech Commands dataset uses a custom HuggingFace loading script, so the extraction script sets `trust_remote_code=True`.

### Quick start (word-level)

```bash
# 1. Extract wav2vec2 features from Speech Commands (35 words × 100 samples)
python scripts/extract_speech_commands.py \
    --output_dir ./acoustic_words_train \
    --limit 100

# 2. Train acoustic model
python scripts/train_acoustic.py \
    --manifest_path ./acoustic_words_train/manifest.json \
    --batch_size 32 \
    --num_epochs 50 \
    --verbose

# 3. Test (auto-detects model from manifest)
python scripts/test_acoustic.py \
    --manifest_path ./acoustic_words_train/manifest.json \
    --verbose
```

### Overfit sanity check

To verify the pipeline works, train on a tiny subset:

```bash
# Extract 10 samples of just "yes"
python scripts/extract_speech_commands.py \
    --output_dir ./acoustic_words_overfit \
    --words yes \
    --limit 10

# Train (should reach 0 errors within ~50 epochs)
python scripts/train_acoustic.py \
    --manifest_path ./acoustic_words_overfit/manifest.json \
    --batch_size 10 \
    --num_epochs 100 \
    --verbose

# Test (expect 100% accuracy)
python scripts/test_acoustic.py \
    --manifest_path ./acoustic_words_overfit/manifest.json \
    --verbose
```

Expected output: `Accuracy: 1.0000 (0.00% error rate)` with predictions exactly matching `Y EH S <EOS>`.

### Model naming

Acoustic models follow the same naming convention as `Ua`:

- `model_name`: `Ua_w2v_LSTM_h128_l1_v42_d0.0_t0.0_s1__gi768`
  - `Ua_w2v` = auditory pathway with wav2vec2 features
  - `gi768` = input dimension (768 for wav2vec2-base)
- `train_name`: `b32_l0.001_fall_s42_sn_ec` (same format as `Ua`)

### Design notes

- **Frozen wav2vec2**: Features are extracted offline (no fine-tuning). Only the projection layer and RNN are trained.
- **Same decoder**: `Ua_w2v` uses the identical `DecoderLSTM` and phoneme vocabulary as `Ua`, enabling direct comparison.
- **Length-safe encoding**: Uses `pack_padded_sequence` so padding doesn't affect hidden state representations.

### Generate learning curve figures

After training, generate learning curve plots from the training logs:

```bash
# Generate figures for all acoustic models
python scripts/make_acoustic_figures.py

# Generate figure for a specific run
python scripts/make_acoustic_figures.py \
    --model_name Ua_w2v_LSTM_h128_l1_v42_d0.0_t0.0_s1__gi768 \
    --train_name b32_l0.001_fall_s42_sn_ec

# Specify output directory
python scripts/make_acoustic_figures.py --output_dir ./figs
```

Figures are saved to `./figs/` by default and include train/valid loss and error curves.

### Run full smoke test

To run the entire acoustic pipeline (extraction → training → testing) for both overfit and full word-level settings:

```bash
./run_acoustic_smoke.sh
```

This creates logs in `./logs_acoustic_words/` and trained models in `./weights/`.

### Load weights

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

## Load weights

Load trained `Ua` (phoneme-input) models:

```python
from swp.utils.models import get_model, load_weights
from swp.utils.setup import set_device

model_name = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
train_name = "b1024_l0.001_fall_s42_sn_ec"
checkpoint = "75"  # epoch number or checkpoint like "1_3"

device = set_device()
model = get_model(model_name)
load_weights(model=model, model_name=model_name, train_name=train_name, checkpoint=checkpoint, device=device)

# model is now ready for evaluation/analysis
```

Alternatively, you can use the test script to evaluate and produce figures for a trained model:

```bash
python scripts/test_repetition.py \
	--model_name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 \
	--train_name b1024_l0.001_fall_s42_sn_ec \
	--checkpoint 75 \
	--batch_size 1024 \
	--verbose
```

Outputs

- Results: `results/evaluation/<model_name>/<train_name>/<checkpoint>/...`
- Figures: `results/figures/<model_name>/<train_name>/<checkpoint>/evaluation/...`

## Repository structure

Top-level folders:

- `swp/` — Core Python package
  - `datasets/` — Dataset loaders and helpers (phoneme folds, evaluation sets)
  - `models/` — Encoders/decoders and container models (auditory Unimodel)
  - `train/` — Training loops (e.g., `repetition.py` saves epoch checkpoints)
  - `test/` — Evaluation logic (behavioral metrics, error analysis)
  - `viz/` — Plotting utilities for figures and analyses
  - `utils/` — Paths, seeding, model name/args parsing, weight IO, grid tools
- `scripts/` — CLI entry points for training/eval and Slurm
  - `train_repetition.py` — Local training (phoneme-input Ua)
  - `train_acoustic.py` — Local training (acoustic-input Ua_w2v)
  - `test_repetition.py` — Evaluate Ua checkpoints; save results + figures
  - `test_acoustic.py` — Evaluate Ua_w2v checkpoints
  - `extract_speech_commands.py` — Extract wav2vec2 features from Speech Commands (word-level)
  - `extract_features.py` — Extract wav2vec2 features from LibriSpeech (sentence-level, experimental)
  - `make_acoustic_figures.py` — Generate learning curve PNGs from training logs
  - `train_repetition.sh` — Slurm training wrapper (Jean Zay)
  - `grid_search.sh` — Submit a grid of Slurm jobs
  - `local_test.sh` — Run Slurm scripts locally (no SBATCH)
  - `generate_queuer.py` — Generate job queue from a grid
- `reproduce/` — Reproduction code for paper figures
  - `scripts/` — Modular analyses (e.g., behavioral, embeddings, ablations, univariate)
  - `reproduce.ipynb` — Unified notebook to re-run figures
  - `datasets/` — CSVs used for reproduction (e.g., WFE, SSP)
- `stimuli/` — Data assets (folds, morphemes, handmade stimuli)
- `weights/` — Local checkpoints directory (auto-created)
- `results/` — Evaluation outputs and figures (auto-created)
- `notebooks/` — Additional analysis notebooks
- `ipa-dict/` — IPA resources
- `CORnet/` — External submodule

Key naming conventions

- `model_name`: Encodes architecture and hyperparameters (decoder type, hidden size, layers, vocab size, dropout, teacher-forcing, start token)
- `train_name`: Encodes training regime (batch size, learning rate, fold, seed, stress flag, loss type)

## Reproduce the paper figures

Run the modular scripts from `reproduce/scripts/` or use the master runner.

Run all analyses via the master script

```bash
cd reproduce/scripts
python run_all.py
```

Run an individual analysis (example: univariate feature importance)

```bash
cd reproduce/scripts
python univariate_analysis.py \
	--model-name Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1 \
	--weights-path weights/1024_75.pth \
	--batch-size 1024 \
	--hidden-size 128
```

Use your own trained weights in analyses

- Pass a direct file path to `--weights-path`, e.g.:
  `weights/<model_name>/<train_name>/<epoch>.pth`
- Keep `--model-name` consistent with how you trained the model.

## Reproducibility practices

- Seeding: we call `seed_everything(42)` and set device consistently (`swp/utils/setup.py`).
- Paths: all save/load locations are centralized in `swp/utils/paths.py`. On Jean Zay, paths switch to `$WORK` automatically; locally they default to the repo folders.
- Caching: scripts save intermediate CSVs/NPYs to `results/` or `reproduce/data/` to avoid recomputation; use `--regenerate` where available to refresh.
- Naming: `model_name` and `train_name` encode configuration and regime for transparent experiments.
- Environment capture: consider exporting `pip freeze > requirements.lock` for archival of your exact environment.

## Troubleshooting

- FileNotFoundError for weights
  - If you run from inside `reproduce/scripts/`, relative paths resolve from that folder. Either run from repo root or pass absolute paths. For trained models, use: `weights/<model_name>/<train_name>/<epoch>.pth`.
- Missing packages (e.g., `sklearn`)
  - Install with `pip install -r requirements.txt`. On Slurm, the job scripts set up the env automatically.
- CUDA not found / GPU not visible
  - Check `python -c "import torch; print(torch.cuda.is_available())"`. On Slurm, ensure the proper module is loaded and `--gres` is set.

## Experimental: Sentence-level acoustic extraction

> **Note**: This section describes sentence-level audio extraction from LibriSpeech, which is **not directly comparable** to the single-word `Ua` models in the paper. Use the Speech Commands word-level pipeline above for comparable experiments.

The `scripts/extract_features.py` script extracts wav2vec2 features from LibriSpeech sentences. This may be useful for future experiments on longer sequences, but the task distribution differs significantly from single-word repetition.

```bash
# Extract features from LibriSpeech validation set (sentences, not comparable to Ua)
python scripts/extract_features.py \
    --output_dir ./acoustic_features_sentences \
    --split validation.clean \
    --limit 100
```

Key differences from word-level:
- LibriSpeech samples are full sentences (50-100+ phonemes)
- Task difficulty is much higher
- Error patterns will differ from single-word models
- Not suitable for direct comparison with paper results

## Citations

Hannagan, T., Agrawal, A., Cohen, L., & Dehaene, A. S. (2021). Emergence of a compositional neural code for written words: Recycling of a convolutional neural network for reading. Proceedings of the National Academy of Sciences, 118(46), e2104779118.

Agrawal, A., & Dehaene, S. (2024). Cracking the neural code for word recognition in convolutional neural networks. arXiv preprint arXiv:2403.06159.

Agrawal, A., & Dehaene, S. (2023). Dissecting the neuronal mechanisms of invariant word recognition. bioRxiv, 2023-11.

Kubilius, J., Schrimpf, M., Nayebi, A., Bear, D., Yamins, D.L.K., DiCarlo, J.J. (2018) CORnet: Modeling the Neural Mechanisms of Core Object Recognition. biorxiv. doi:10.1101/408385

Kubilius, J., Schrimpf, M., Kar, K., Rajalingham, R., Hong, H., Majaj, N., ... & Dicarlo, J. (2019). Brain-like object recognition with high-performing shallow recurrent ANNs. In Advances in Neural Information Processing Systems (pp. 12785-12796).

Burgess, N., & Hitch, G. J. (1992). Toward a network model of the articulatory loop. Journal of memory and language, 31(4), 429-460.

Botvinick, M. M., & Plaut, D. C. (2006). Short-term memory for serial order: a recurrent neural network model. Psychological review, 113(2), 201.

Sajid, N., Holmes, E., Costa, L. D., Price, C., & Friston, K. (2022). A mixed generative model of auditory word repetition. bioRxiv, 2022-01.
