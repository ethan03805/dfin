#!/usr/bin/env bash
# Full Phase 0 verification on this machine: the tiny test models, workspace tests,
# release build, and the monolithic / in-process split / local-pod benchmark on each
# tiny model.
#
# Needs the patched llama.cpp build (scripts/apply-llama-patch.sh, scripts/build-llama.sh)
# and a Python with numpy and gguf (`pip install numpy vendor/llama.cpp/gguf-py`; the
# project venv is used when present). Tests and benchmarks use the release profile.
set -u
cd "$(dirname "$0")/.."
. scripts/common.sh

echo "### tiny test models (generated when missing)"
mkdir -p models
gen() {
    local out="models/$1" tool="$2"
    shift 2
    if [ -f "$out" ]; then
        echo "have $out"
    else
        "$PY" "tools/$tool" "$out" "$@" || exit 1
    fi
}
# What the workspace tests load.
gen tiny-llama-f32.gguf make_tiny_llama.py
gen tiny-kimi-k3-res.gguf make_tiny_kimi_k3.py --attn-res-block-size 2
gen tiny-glm5-next.gguf make_tiny_glm5_next.py
if [ -f models/tiny-shards/tiny-llama-f32.manifest.json ]; then
    echo "have models/tiny-shards"
else
    "$PY" tools/shard_gguf.py models/tiny-llama-f32.gguf --out models/tiny-shards | tail -1 || exit 1
fi
# What only the benchmark below loads.
gen tiny-glm-dsa-f32.gguf make_tiny_glm_dsa.py
gen tiny-kimi-k3-f32.gguf make_tiny_kimi_k3.py

echo "### cargo test --release --workspace"
cargo test --release --workspace --locked 2>&1 | grep -E "^test |test result|Running|error|panicked|SKIP"
echo "### cargo build --release"
cargo build --release --locked 2>&1 | tail -2
need_dfin
for spec in "tiny-llama-f32.gguf 3,6 0" "tiny-llama-f32.gguf 3,6 -1" "tiny-glm-dsa-f32.gguf 2,6 0" "tiny-kimi-k3-f32.gguf 2,5 0"; do
    set -- $spec
    echo "### bench $1 split $2 ngl $3"
    "$B" --network direct bench --model "models/$1" --split "$2" --gen-tokens 64 --prompt-tokens 32 --n-gpu-layers "$3" --log warn 2>&1 | grep -vE "^\s*$" | tail -12
done
echo "### done"
