#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

LOGDIR="./logs_acoustic_words"
mkdir -p "$LOGDIR"

OVERDIR="./acoustic_words_overfit"
TRAINDIR="./acoustic_words_train"

echo "== [0/6] Sanity: scripts exist =="
test -f scripts/extract_speech_commands.py
test -f scripts/train_acoustic.py
test -f scripts/test_acoustic.py

echo "== [1/6] Extract Speech Commands (overfit: 10 samples of 'yes') =="
python scripts/extract_speech_commands.py \
  --output_dir "$OVERDIR" \
  --words yes \
  --limit 10 2>&1 | tee "$LOGDIR/01_extract_overfit.log"

OVER_MANIFEST="$OVERDIR/manifest.json"
if [ ! -f "$OVER_MANIFEST" ]; then
  echo "ERROR: manifest not found at $OVER_MANIFEST"
  find "$OVERDIR" -maxdepth 3 -type f | head -n 50
  exit 1
fi
echo "Using overfit manifest: $OVER_MANIFEST"

echo "== [2/6] Train (overfit) =="
python scripts/train_acoustic.py \
  --manifest_path "$OVER_MANIFEST" \
  --batch_size 10 \
  --num_epochs 100 \
  --learn_rate 0.001 \
  --verbose 2>&1 | tee "$LOGDIR/02_train_overfit.log"

echo "== [3/6] Test (overfit) =="
# Auto-detect finds the run whose run_info.json manifest_path matches $OVER_MANIFEST
python scripts/test_acoustic.py \
  --manifest_path "$OVER_MANIFEST" \
  --verbose 2>&1 | tee "$LOGDIR/03_test_overfit.log"

echo "== [4/6] Extract Speech Commands (full word-level: limit 100) =="
python scripts/extract_speech_commands.py \
  --output_dir "$TRAINDIR" \
  --limit 100 2>&1 | tee "$LOGDIR/04_extract_full.log"

TRAIN_MANIFEST="$TRAINDIR/manifest.json"
if [ ! -f "$TRAIN_MANIFEST" ]; then
  echo "ERROR: manifest not found at $TRAIN_MANIFEST"
  find "$TRAINDIR" -maxdepth 3 -type f | head -n 50
  exit 1
fi
echo "Using train manifest: $TRAIN_MANIFEST"

echo "== [5/6] Train (full word-level) =="
python scripts/train_acoustic.py \
  --manifest_path "$TRAIN_MANIFEST" \
  --batch_size 32 \
  --num_epochs 50 \
  --learn_rate 0.001 \
  --verbose 2>&1 | tee "$LOGDIR/05_train_full.log"

echo "== [6/6] Test (full word-level) =="
# Auto-detect finds the run whose run_info.json manifest_path matches $TRAIN_MANIFEST
python scripts/test_acoustic.py \
  --manifest_path "$TRAIN_MANIFEST" \
  --verbose 2>&1 | tee "$LOGDIR/06_test_full.log"

echo "DONE ✅ Logs in $LOGDIR"
