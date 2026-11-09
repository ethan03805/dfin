#!/usr/bin/env bash
# Real-model check of the pipeline on one machine.
#
#   scripts/real-model-bench.sh <model.gguf> <split, e.g. 16,32> [gen_tokens] [prompt_tokens]
#
# Runs, in order:
#   1. GGUF metadata (architecture, layers, width, experts)
#   2. stock llama.cpp baseline with llama-bench (GPU, experts in RAM)
#   3. dfin bench: monolithic / in-process split / local pod over QUIC, GPU + experts in RAM, f16 wire
#   4. the same with f32 on the wire (exactness)
#   5. the same CPU-only, fewer tokens
#   6. a live pod with per-piece memory sampling and one chat request through the OpenAI API
#
# Runs on Linux, macOS and Windows Git Bash (scripts/common.sh picks the binary, curl
# and the process tools). Steps 1 and 2 need the llama.cpp tools, which a
# TARGETS=llama build leaves out; they are skipped when absent.
set -u
cd "$(dirname "$0")/.."
. scripts/common.sh
MODEL="${1:?model path}"
SPLIT="${2:?split, e.g. 16,32}"
GEN="${3:-64}"
PROMPT="${4:-256}"
need_dfin
# `dfin pod` starts its client without --model-name, so the client serves the file stem.
NAME="$(model_name "$MODEL")"
LB="$LLAMA_BIN"
filter() { grep -vE "^\s*$|^llama_model_loader|^print_info|^load_tensors|^llama_context|^llama_kv|^graph_reserve|^load:|^sched_reserve|^llama_graph|^llama_memory|^llama_rs|~llama_context|^load_hparams|^\.\.\.\."; }

echo "### 1. metadata"
if [ ! -x "$LB/llama-gguf$EXE" ]; then
    echo "skipped: no $LB/llama-gguf$EXE"
else
"$LB/llama-gguf$EXE" "$MODEL" r n 2>&1 | grep -E "general.architecture|general.name|block_count|embedding_length|expert_count|expert_used_count|nextn|head_count|context_length|file_type|chat_template" | grep -v "tokenizer.chat_template = " | head -20
fi

echo "### 2. stock llama.cpp baseline (llama-bench, -ngl 99, experts on CPU)"
if [ ! -x "$LB/llama-bench$EXE" ]; then
    echo "skipped: no $LB/llama-bench$EXE"
else
"$LB/llama-bench$EXE" -m "$MODEL" -ngl 99 -ot "exps=CPU" -p "$PROMPT" -n "$GEN" -r 2 2>&1 | grep -E "^\||model|error" | tail -6
fi

echo "### 3. dfin bench, GPU + experts in RAM, f16 wire"
"$B" --network direct bench --model "$MODEL" --split "$SPLIT" --gen-tokens "$GEN" --prompt-tokens "$PROMPT" --n-gpu-layers -1 --cpu-tensors exps --act-dtype f16 --log warn 2>&1 | filter | tail -9

echo "### 4. dfin bench, GPU + experts in RAM, f32 wire"
"$B" --network direct bench --model "$MODEL" --split "$SPLIT" --gen-tokens "$GEN" --prompt-tokens "$PROMPT" --n-gpu-layers -1 --cpu-tensors exps --act-dtype f32 --log warn 2>&1 | filter | tail -9

echo "### 5. dfin bench, CPU only, f16 wire"
"$B" --network direct bench --model "$MODEL" --split "$SPLIT" --gen-tokens 32 --prompt-tokens 128 --n-gpu-layers 0 --act-dtype f16 --log warn 2>&1 | filter | tail -9

echo "### 6. live pod: memory per piece and one chat request"
PORT=8123
"$B" --network direct pod --model "$MODEL" --split "$SPLIT" --listen 127.0.0.1:$PORT --base-port 7201 --n-ctx 8192 --n-seq-max 2 --n-gpu-layers -1 --cpu-tensors exps --act-dtype f16 --log info > pod.log 2>&1 &
PODPID=$!
for i in $(seq 1 240); do
    if "$CURL" -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
done
echo "pod ready after ${i}s"
echo "--- memory per process (MB) and GPU/CPU buffer sizes from the logs ---"
dfin_memory
grep -E "model buffer size|KV buffer size|compute buffer size|loading head|loading tail|stage ready|partial model" pod.log | strip_ansi | cut -c1-160 | head -30
echo "--- chat request ---"
"$CURL" -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "content-type: application/json" -d '{"model":"'"$NAME"'","messages":[{"role":"user","content":"In two sentences, what is pipeline parallelism in neural network inference? /nothink"}],"max_tokens":120,"temperature":0}' | head -c 2000; echo
echo "--- second request, same prefix plus a follow-up (prefix cache) ---"
"$CURL" -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "content-type: application/json" -d '{"model":"'"$NAME"'","messages":[{"role":"user","content":"In two sentences, what is pipeline parallelism in neural network inference? /nothink"},{"role":"assistant","content":"It splits the layers of a model across several devices."},{"role":"user","content":"Name one drawback. /nothink"}],"max_tokens":60,"temperature":0}' | head -c 1500; echo
echo "--- /status ---"
"$CURL" -s "http://127.0.0.1:$PORT/status" | head -c 3000; echo
stop_pids $PODPID; stop_all_dfin
echo "### done"
