#!/usr/bin/env bash
# Follow-up to real-model-bench.sh: is the extra per-token cost of a local pod (several
# processes on one CPU) thread contention? Re-run the pod benchmark with (a) passive
# OpenMP waiting and (b) fewer threads per piece, then sample memory per process.
#
#   scripts/real-model-contention.sh <model.gguf> <split>
set -u
cd "$(dirname "$0")/.."
MODEL="${1:?model path}"; SPLIT="${2:?split}"
B=target/release/dfin.exe; [ -f "$B" ] || B=target/release/dfin
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
for i in $(seq 1 240); do curl.exe -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
echo "pod ready after ${i}s"
# one request so KV and compute buffers are allocated
curl.exe -s "http://127.0.0.1:$PORT/v1/completions" -H "content-type: application/json" -d '{"prompt":"The three laws of robotics are","max_tokens":16,"temperature":0}' > /dev/null
powershell.exe -NoProfile -Command "Get-Process dfin | Sort-Object Id | ForEach-Object { '{0,8}  WS {1,7:N0} MB  private {2,7:N0} MB  cmd: {3}' -f \$_.Id, (\$_.WorkingSet64/1MB), (\$_.PrivateMemorySize64/1MB), ((Get-CimInstance Win32_Process -Filter ('ProcessId=' + \$_.Id)).CommandLine -replace '.*dfin(\.exe)? ','' -replace ' --model \S+','' -replace ' --listen \S+','' -replace ' --n-ctx.*','') }"
echo "--- model sizes per piece (from logs) ---"
sed 's/\x1b\[[0-9;]*m//g' pod-mem.log | grep -iE "stage ready|model loaded|size|layers=" | cut -c1-200 | head -12
powershell.exe -NoProfile -Command "Get-CimInstance Win32_OperatingSystem | ForEach-Object { 'system RAM: total {0:N0} MB, free {1:N0} MB' -f (\$_.TotalVisibleMemorySize/1KB), (\$_.FreePhysicalMemory/1KB) }"
kill $PODPID 2>/dev/null; sleep 2; taskkill /F /IM dfin.exe >/dev/null 2>&1
echo "### done"
