#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT}"
mkdir -p "$DATA_ROOT/datasets/raw"
raw="$DATA_ROOT/datasets/raw/cube_single_expert.tar.zst"
expected=3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600
if [[ ! -s "$raw" ]]; then
  curl --fail --location --connect-timeout 15 --speed-time 120 --speed-limit 1024 \
    --retry 12 --retry-all-errors --continue-at - \
    "${HF_DATASETS_ENDPOINT:-https://huggingface.co/datasets}/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst" \
    -o "$raw.part"
  mv "$raw.part" "$raw"
fi
printf '%s  %s\n' "$expected" "$raw" | sha256sum -c -
tar --zstd -xf "$raw" -C "$DATA_ROOT/datasets"
test -s "$DATA_ROOT/datasets/cube_single_expert.h5"
printf '%s  %s\n' "$expected" cube_single_expert.tar.zst > "$DATA_ROOT/datasets/cube_source_sha256.txt"
touch "$DATA_ROOT/datasets/cube.ready"
echo "cube_ready_utc=$(date -u +%FT%TZ)"
