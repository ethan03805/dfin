#!/usr/bin/env bash
# Apply the layer-range patch to the vendored llama.cpp checkout (idempotent).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root/vendor/llama.cpp"
if [ ! -f include/llama.h ]; then
    echo "vendor/llama.cpp is empty; run: git submodule update --init --depth 1" >&2
    exit 1
fi
if grep -q "llama_model_layer_end" include/llama.h; then
    echo "layer-range patch already applied"
    exit 0
fi
git apply --check ../patches/0001-layer-range.patch
git apply ../patches/0001-layer-range.patch
echo "layer-range patch applied to $(git rev-parse --short HEAD)"
