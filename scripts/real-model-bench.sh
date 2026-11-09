#!/usr/bin/env bash
# Real-model check of the pipeline on one machine.
#
#   scripts/real-model-bench.sh <model.gguf> <split, e.g. 16,32> [gen_tokens] [prompt_tokens]
#
# Runs, in order:
#   1. GGUF metadata (architecture, layers, width, experts)
#   2. stock llama.cpp baseline with llama-bench (GPU, experts in RAM)
#   3. dfin bench: monolithic / in-process split / local pod over TCP, GPU + experts in RAM, f16 wire
#   4. the same with f32 on the wire (exactness)
#   5. the same CPU-only, fewer tokens
#   6. a live pod with per-piece memory sampling and one chat request through the OpenAI API
set -u
cd "$(dirname "$0")/.."
MODEL="${1:?model path}"
SPLIT="${2:?split, e.g. 16,32}"
GEN="${3:-64}"
PROMPT="${4:-256}"
B=target/release/dfin.exe; [ -f "$B" ] || B=target/release/dfin
LB=vendor/llama.cpp/build/bin/Release
filter() { grep -vE "^\s*$|^llama_model_loader|^print_info|^load_tensors|^llama_context|^llama_kv|^graph_reserve|^load:|^sched_reserve|^llama_graph|^llama_memory|^llama_rs|~llama_context|^load_hparams|^\.\.\.\."; }

echo "### 1. metadata"
"$LB/llama-gguf.exe" "$MODEL" r n 2>&1 | grep -E "general.architecture|general.name|block_count|embedding_length|expert_count|expert_used_count|nextn|head_count|context_length|file_type|chat_template" | grep -v "tokenizer.chat_template = " | head -20

echo "### 2. stock llama.cpp baseline (llama-bench, -ngl 99, experts on CPU)"
"$LB/llama-bench.exe" -m "$MODEL" -ngl 99 -ot "exps=CPU" -p "$PROMPT" -n "$GEN" -r 2 2>&1 | grep -E "^\||model|error" | tail -6

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
    if curl.exe -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
done
echo "pod ready after ${i}s"
echo "--- per-process working set (MB) and Vulkan/CPU buffer sizes from the logs ---"
tasklist /FI "IMAGENAME eq dfin.exe" /FO CSV 2>/dev/null | tail -n +2 | awk -F'","' '{gsub(/[",]/,"",$5); printf "  dfin pid %s: %.0f MB\n", $2, $5/1024}'
grep -E "model buffer size|KV buffer size|compute buffer size|loading head|loading tail|stage ready|partial model" pod.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-160 | head -30
echo "--- chat request ---"
curl.exe -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "content-type: application/json" -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"In two sentences, what is pipeline parallelism in neural network inference? /nothink"}],"max_tokens":120,"temperature":0}' | head -c 2000; echo
echo "--- second request, same prefix plus a follow-up (prefix cache) ---"
curl.exe -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "content-type: application/json" -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"In two sentences, what is pipeline parallelism in neural network inference? /nothink"},{"role":"assistant","content":"It splits the layers of a model across several devices."},{"role":"user","content":"Name one drawback. /nothink"}],"max_tokens":60,"temperature":0}' | head -c 1500; echo
echo "--- /status ---"
curl.exe -s "http://127.0.0.1:$PORT/status" | head -c 3000; echo
kill $PODPID 2>/dev/null; sleep 2; taskkill /F /IM dfin.exe >/dev/null 2>&1
echo "### done"
