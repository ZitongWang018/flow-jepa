#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DATA_ROOT/checkpoints/flow-jepa/lewm-initialization}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p "$CHECKPOINT_DIR"

download() {
  local repo="$1"
  local source="$2"
  local dest="$3"
  curl --fail --location --retry 8 --retry-all-errors \
    "$HF_ENDPOINT/$repo/resolve/main/$source" \
    -o "$CHECKPOINT_DIR/$dest.part"
  mv "$CHECKPOINT_DIR/$dest.part" "$CHECKPOINT_DIR/$dest"
}

download quentinll/lewm-pusht weights.pt lewm-pusht-weights.pt
download quentinll/lewm-pusht config.json lewm-pusht-config.json
download quentinll/lewm-reacher weights.pt lewm-reacher-weights.pt
download quentinll/lewm-reacher config.json lewm-reacher-config.json

(
  cd "$CHECKPOINT_DIR"
  sha256sum *.pt *.json | tee SHA256SUMS
)
