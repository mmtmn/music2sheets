#!/usr/bin/env bash
# Fetch hFT-Transformer (Toyama et al., ISMIR 2023; MIT licence) and its MAESTRO-V3 checkpoint into
# third_party/hFT-Transformer: the optional third piano model (m2s/transcribe/hft.py). Pinned to a
# commit and the `ismir2023` release asset, checked by SHA-256. Enable it with M2S_HFT=1.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=https://github.com/sony/hFT-Transformer.git
COMMIT=71a2ee06e9ced1ea24673c95ee0acded2fc98d04
ZIP_URL=https://github.com/sony/hFT-Transformer/releases/download/ismir2023/checkpoint.zip
ZIP_SHA256=bb3c88758e4802e6b3881ccb6dc7927417fb43f76980df3739ab2bbc418d70c5
DIR=third_party/hFT-Transformer

mkdir -p third_party
if [ ! -d "$DIR/.git" ]; then
  git clone "$REPO" "$DIR"
fi
git -C "$DIR" checkout --quiet "$COMMIT"
if [ ! -f "$DIR/checkpoint/MAESTRO-V3/model_016_003.pkl" ]; then
  curl -fL -o "$DIR/checkpoint.zip" "$ZIP_URL"
  echo "$ZIP_SHA256  $DIR/checkpoint.zip" | sha256sum -c -
  unzip -o -q "$DIR/checkpoint.zip" -d "$DIR"
fi
echo "hFT-Transformer ready in $DIR (use with M2S_HFT=1)"
