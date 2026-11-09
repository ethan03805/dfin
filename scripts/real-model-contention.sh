#!/usr/bin/env bash
# Follow-up to real-model-bench.sh: is the extra per-token cost of a local pod (several
# processes on one CPU) thread contention? Re-run the pod benchmark with (a) passive
# OpenMP waiting and (b) fewer threads per piece, then sample memory per process.
#
#   scripts/real-model-contention.sh <model.gguf> <split>
#
# Runs on Linux, macOS and Windows Git Bash (scripts/common.sh picks the binary, curl
# and the process tools).
set -u
cd "$(dirname "$0")/.."
. scripts/common.sh
MODEL="${1:?model path}"; SPLIT="${2:?split}"
need_dfin
filter() { grep -E "^\| [123] "; }

echo "### A. default threads, OMP_WAIT_POLICY=PASSIVE"
OMP_WAIT_POLICY=PASSIVE "$B" --network direct bench --model "$MODEL" --split "$SPLIT" --gen-tokens 64 --prompt-tokens 256 --n-gpu-layers -1 --cpu-tensors exps --act-dtype f16 --log warn 2>&1 | filter
echo "### B. --threads 3 per piece"
"$B" --network direct bench --model "$MODEL" --split "$SPLIT" --gen-tokens 64 --prompt-tokens 256 --n-gpu-layers -1 --cpu-tensors exps --act-dtype f16 --threads 3 --log warn 2>&1 | filter
echo "### C. --threads 3 and OMP_WAIT_POLICY=PASSIVE"
OMP_WAIT_POLICY=PASSIVE "$B" --network direct bench --model "$MODEL" --split "$SPLIT" --gen-tokens 64 --prompt-tokens 256 --n-gpu-layers -1 --cpu-tensors exps --act-dtype f16 --threads 3 --log warn 2>&1 | filter

echo "### D. memory per piece (GPU + experts in RAM)"
PORT=8124
"$B" --network direct pod --model "$MODEL" --split "$SPLIT" --listen 127.0.0.1:$PORT --base-port 7301 --n-ctx 8192 --n-seq-max 2 --n-gpu-layers -1 --cpu-tensors exps --act-dtype f16 --log info > pod-mem.log 2>&1 &
PODPID=$!
for i in $(seq 1 240); do "$CURL" -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
echo "pod ready after ${i}s"
# one request so KV and compute buffers are allocated
"$CURL" -s "http://127.0.0.1:$PORT/v1/completions" -H "content-type: application/json" -d '{"prompt":"The three laws of robotics are","max_tokens":16,"temperature":0}' > /dev/null
dfin_memory
echo "--- model sizes per piece (from logs) ---"
strip_ansi < pod-mem.log | grep -iE "stage ready|model loaded|size|layers=" | cut -c1-200 | head -12
system_memory
stop_pids $PODPID; stop_all_dfin
echo "### done"
