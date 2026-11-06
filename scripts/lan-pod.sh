#!/usr/bin/env bash
# Two-machine LAN pod: a remote stage already running on another machine serves layers
# 1:N; this machine runs the head (layer 0), a local stage for N:LAST, and the tail.
#
#   scripts/lan-pod.sh <model.gguf> <remote host:port> <N> <LAST> [gen_tokens] [prompt_tokens]
#
# e.g. scripts/lan-pod.sh models/glm-4.7-flash/zai-org_GLM-4.7-Flash-Q4_K_M.gguf 10.0.0.38:7001 21 46
set -u
cd "$(dirname "$0")/.."
MODEL="${1:?model}"; REMOTE="${2:?host:port}"; N="${3:?N}"; LAST="${4:?LAST}"
GEN="${5:-64}"; PROMPT="${6:-256}"
B=target/release/dfin.exe; [ -f "$B" ] || B=target/release/dfin
API=127.0.0.1:8080
strip() { sed 's/\x1b\[[0-9;]*m//g'; }

echo "### remote stage reachability"
HOST="${REMOTE%%:*}"
ping -n 10 "$HOST" 2>/dev/null | tail -2 || ping -c 10 "$HOST" | tail -2

echo "### local stage $N:$LAST"
"$B" stage --model "$MODEL" --layers "$N:$LAST" --listen 127.0.0.1:7002 --n-gpu-layers -1 --cpu-tensors exps ${STAGE_ARGS:-} \
    --n-ctx 8192 --n-seq-max 2 --act-dtype f16 --log info > lan-stage.log 2>&1 &
STAGEPID=$!
for i in $(seq 1 300); do (exec 3<>/dev/tcp/127.0.0.1/7002) 2>/dev/null && break; sleep 1; done
echo "local stage up after ${i}s"

echo "### client: head 0:1 -> $REMOTE=1:$N -> 127.0.0.1:7002=$N:$LAST -> tail $LAST:end"
"$B" client --model "$MODEL" --head 0:1 --tail "$LAST:end" --stage "$REMOTE=1:$N" --stage "127.0.0.1:7002=$N:$LAST" \
    --listen $API --n-gpu-layers -1 --cpu-tensors exps --max-sessions 2 --n-ctx 8192 --act-dtype f16 --log info > lan-client.log 2>&1 &
CLIENTPID=$!
for i in $(seq 1 180); do curl.exe -s "http://$API/health" >/dev/null 2>&1 && break; sleep 1; done
echo "client up after ${i}s"
strip < lan-client.log | grep -E "connected to stage|stage ping|engine ready" | cut -c1-200

if [ "${QUICK:-0}" = "1" ]; then
    echo "### QUICK: one 256-token completion through the LAN pipeline"
    curl.exe -s "http://$API/v1/completions" -H "content-type: application/json"         -d "{\"prompt\":\"$(printf 'The quick brown fox jumps over the lazy dog while the patient owl watches from the old oak tree. %.0s' $(seq 1 14))\",\"max_tokens\":32,\"temperature\":0}"         | /w/.venv/Scripts/python.exe -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['usage']))"
    curl.exe -s "http://$API/status" | /w/.venv/Scripts/python.exe -c "import sys,json; d=json.load(sys.stdin); s=d['stats']; print(json.dumps({k:v for k,v in s.items() if k!='stages'})); [print(json.dumps(x)) for x in s['stages']]"
    kill $CLIENTPID $STAGEPID 2>/dev/null; sleep 2
    echo "### done (quick)"
    exit 0
fi
echo "### bench: local monolithic, local in-process split, remote pipeline"
"$B" bench --model "$MODEL" --split "1,$N,$LAST" --prompt-tokens "$PROMPT" --gen-tokens "$GEN" --n-gpu-layers -1 --cpu-tensors exps \
    --act-dtype f16 --api-url "http://$API" --log warn 2>&1 | strip | grep -E "^\| |^model:|^layer-range"

echo "### a longer generation through the LAN pipeline (128 tokens)"
curl.exe -s "http://$API/v1/completions" -H "content-type: application/json" \
    -d '{"prompt":"The history of the Roman Republic begins","max_tokens":128,"temperature":0}' \
    | /w/.venv/Scripts/python.exe -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['usage'])); print(d['choices'][0]['text'][:400])"

echo "### chat, two turns"
curl.exe -s "http://$API/v1/chat/completions" -H "content-type: application/json" \
    -d '{"model":"glm","messages":[{"role":"user","content":"In two sentences, why is network latency the main cost when a model is split across homes? /nothink"}],"max_tokens":120,"temperature":0}' \
    | /w/.venv/Scripts/python.exe -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print(json.dumps(d['usage']))"
curl.exe -s "http://$API/v1/chat/completions" -H "content-type: application/json" \
    -d '{"model":"glm","messages":[{"role":"user","content":"In two sentences, why is network latency the main cost when a model is split across homes? /nothink"},{"role":"assistant","content":"Every generated token must cross each link in turn."},{"role":"user","content":"How would speculative decoding help? One sentence. /nothink"}],"max_tokens":80,"temperature":0}' \
    | /w/.venv/Scripts/python.exe -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print(json.dumps(d['usage']))"

echo "### /status"
curl.exe -s "http://$API/status" | /w/.venv/Scripts/python.exe -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['prefix_cache'])); s=d['stats']; print(json.dumps({k:v for k,v in s.items() if k!='stages'})); [print(json.dumps(x)) for x in s['stages']]"

kill $CLIENTPID $STAGEPID 2>/dev/null; sleep 2
echo "### local processes stopped; remote stage left running"
