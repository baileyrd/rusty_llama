#!/usr/bin/env bash
#
# Fetch the tiny "TinyStories" model + the Llama tokenizer used by rusty_llama.
#
#   ./scripts/download_assets.sh                # stories15M (~60 MB)
#   ./scripts/download_assets.sh stories42M     # bigger sibling
#   ./scripts/download_assets.sh stories110M    # bigger still
#
# Models come from Karpathy's HuggingFace repo; the tokenizer from the llama2.c
# GitHub repo. (HuggingFace may be blocked in some sandboxes — run this on a
# machine with open network access.)
set -euo pipefail

MODEL="${1:-stories15M}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo ">> tokenizer.bin"
curl -fL --retry 3 -o tokenizer.bin \
  https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin

echo ">> ${MODEL}.bin"
curl -fL --retry 3 -o "${MODEL}.bin" \
  "https://huggingface.co/karpathy/tinyllamas/resolve/main/${MODEL}.bin"

cat <<EOF

Done. Try:
  cargo run --release -- ${MODEL}.bin -z tokenizer.bin -i "Once upon a time" -n 256
EOF
