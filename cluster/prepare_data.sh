#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
DATASET_DIR="$DATA_ROOT/datasets"
RAW="$DATASET_DIR/raw"
HF_DATASETS_ENDPOINT="${HF_DATASETS_ENDPOINT:-https://huggingface.co/datasets}"
mkdir -p "$RAW" "$DATASET_DIR"

download() {
  local repo="$1" file="$2"
  local dest="$RAW/$file"
  local url="$HF_DATASETS_ENDPOINT/$repo/resolve/main/$file"
  if [[ -f "$dest" && -s "$dest" ]]; then
    echo "EXISTS $file $(stat -c %s "$dest")"
  else
    echo "START $file"
    curl --fail --location --retry 12 --retry-all-errors --continue-at - \
      "$url" -o "$dest.part"
    mv "$dest.part" "$dest"
    echo "DONE $file $(stat -c %s "$dest")"
  fi
}

download quentinll/lewm-pusht pusht_expert_train.h5.zst &
download quentinll/lewm-tworooms tworoom.tar.zst &
download quentinll/lewm-reacher reacher.tar.zst &
download quentinll/lewm-cube cube_single_expert.tar.zst &
wait

(
  cd "$RAW"
  sha256sum pusht_expert_train.h5.zst tworoom.tar.zst reacher.tar.zst cube_single_expert.tar.zst \
    | tee "$DATASET_DIR/source_sha256.txt"
)
grep -E '^7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318  pusht_expert_train.h5.zst$' "$DATASET_DIR/source_sha256.txt"
grep -E '^494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca  tworoom.tar.zst$' "$DATASET_DIR/source_sha256.txt"
grep -E '^4ff2385e49712caa89f21b8e0a246e2614b621d3f22cf2d1224d845e879a1cc2  reacher.tar.zst$' "$DATASET_DIR/source_sha256.txt"
grep -E '^3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600  cube_single_expert.tar.zst$' "$DATASET_DIR/source_sha256.txt"

zstd -d -f "$RAW/pusht_expert_train.h5.zst" -o "$DATASET_DIR/pusht_expert_train.h5"
tar --zstd -xf "$RAW/tworoom.tar.zst" -C "$DATASET_DIR"
tar --zstd -xf "$RAW/reacher.tar.zst" -C "$DATASET_DIR"
tar --zstd -xf "$RAW/cube_single_expert.tar.zst" -C "$DATASET_DIR"

find "$DATASET_DIR" -maxdepth 2 -type f -name '*.h5' -printf '%p %s\n' | sort
