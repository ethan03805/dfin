#!/usr/bin/env bash
# Phase 1 chain check on a real model, one machine: monolithic, in-process split, and a
# local pod of two stages driven through the chain in direct and reverse return modes.
#
#   scripts/phase1-chain-bench.sh <model.gguf> [split] [gen_tokens] [prompt_tokens]
set -u
cd "$(dirname "$0")/.."
MODEL="${1:?model}"; SPLIT="${2:-1,24,46}"; GEN="${3:-64}"; PROMPT="${4:-256}"
B=target/release/dfin.exe; [ -f "$B" ] || B=target/release/dfin
strip() { sed 's/\x1b\[[0-9;]*m//g'; }
for mode in direct reverse; do
    echo "### return mode: $mode"
    "$B" --network direct bench --model "$MODEL" --split "$SPLIT" --prompt-tokens "$PROMPT" --gen-tokens "$GEN" \
        --n-gpu-layers -1 --cpu-tensors exps --act-dtype f16 --return "$mode" --log warn 2>&1 \
        | strip | grep -E "^\| |^model:|^layer-range"
done
echo "### done"
