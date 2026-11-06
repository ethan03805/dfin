#!/usr/bin/env bash
# Full Phase 0 verification on this machine: workspace tests, release build, and the
# monolithic / in-process split / local-pod-over-TCP benchmark on each tiny model.
set -u
cd "$(dirname "$0")/.."
echo "### cargo test --workspace"
cargo test --workspace 2>&1 | grep -E "^test |test result|Running|error|panicked|SKIP" 
echo "### cargo build --release"
cargo build --release 2>&1 | tail -2
B=target/release/dfin.exe
[ -f "$B" ] || B=target/release/dfin
for spec in "tiny-llama-f32.gguf 3,6 0" "tiny-llama-f32.gguf 3,6 -1" "tiny-glm-dsa-f32.gguf 2,6 0" "tiny-kimi-k3-f32.gguf 2,5 0"; do
    set -- $spec
    echo "### bench $1 split $2 ngl $3"
    "$B" bench --model "models/$1" --split "$2" --gen-tokens 64 --prompt-tokens 32 --n-gpu-layers "$3" --log warn 2>&1 | grep -vE "^\s*$" | tail -12
done
echo "### done"
