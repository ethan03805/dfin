#!/usr/bin/env bash
# Two-machine LAN pod: a remote stage already running on another machine serves layers
# 1:N; this machine runs the head (layer 0), a local stage for N:LAST, and the tail.
#
#   scripts/lan-pod.sh <model.gguf> <remote host:port/node-id> <N> <LAST> [gen_tokens] [prompt_tokens]
#
# The remote stage prints its node id at startup. RETURN=direct|reverse (default direct)
# picks how results come back; RELAY=host:port/node-id routes through a relay instead.
#
# Stages dial each other, so the remote stage must be able to reach the local one:
# set LOCAL_STAGE=<this machine's LAN address>:7002. The default, 127.0.0.1:7002, only
# works when the "remote" stage runs on this machine too.
#
# e.g. scripts/lan-pod.sh models/glm-4.7-flash/zai-org_GLM-4.7-Flash-Q4_K_M.gguf 10.0.0.38:7001/<id> 21 46
#
# Runs on Linux, macOS and Windows Git Bash (scripts/common.sh picks the binary, Python
# and curl). The client is started with --model-name, and the chat requests name it.
set -u
cd "$(dirname "$0")/.."
. scripts/common.sh
MODEL="${1:?model}"; REMOTE="${2:?host:port}"; N="${3:?N}"; LAST="${4:?LAST}"
GEN="${5:-64}"; PROMPT="${6:-256}"
need_dfin
NAME="$(model_name "$MODEL")"
API=127.0.0.1:8080
LOCAL_STAGE="${LOCAL_STAGE:-127.0.0.1:7002}"
case "$LOCAL_STAGE" in
    127.*) LISTEN="$LOCAL_STAGE" ;;
    *) LISTEN="0.0.0.0:${LOCAL_STAGE##*:}" ;;
esac
strip() { strip_ansi; }

echo "### remote stage reachability"
HOST="${REMOTE%%:*}"
case "$HOST:$LOCAL_STAGE" in
    127.*) ;;
    *:127.*) echo "warning: $HOST cannot dial the local stage at $LOCAL_STAGE; set LOCAL_STAGE to a LAN address" >&2 ;;
esac
ping_host "$HOST" 2>/dev/null | tail -2

echo "### local stage $N:$LAST"
KEY=lan-stage.key
"$B" --network direct stage --model "$MODEL" --layers "$N:$LAST" --listen "$LISTEN" --identity "$KEY" --n-gpu-layers -1 --cpu-tensors exps ${STAGE_ARGS:-} \
    --n-ctx 8192 --n-seq-max 2 --act-dtype f16 --log info > lan-stage.log 2>&1 &
STAGEPID=$!
for i in $(seq 1 300); do grep -q "stage ready" lan-stage.log 2>/dev/null && break; sleep 1; done
echo "local stage up after ${i}s"
LOCAL_ID=$(strip < lan-stage.log | sed -n 's/.*node id \([0-9a-f]\{64\}\).*/\1/p' | head -1)

RET_ARGS="--return ${RETURN:-direct}"
[ -n "${RELAY:-}" ] && RET_ARGS="--relay $RELAY"
echo "### client: head 0:1 -> $REMOTE=1:$N -> $LOCAL_STAGE/$LOCAL_ID=$N:$LAST -> tail $LAST:end ($RET_ARGS)"
"$B" --network direct client --model "$MODEL" --head 0:1 --tail "$LAST:end" --stage "$REMOTE=1:$N" --stage "$LOCAL_STAGE/$LOCAL_ID=$N:$LAST" $RET_ARGS \
    --listen $API --model-name "$NAME" --n-gpu-layers -1 --cpu-tensors exps --max-sessions 2 --n-ctx 8192 --act-dtype f16 --log info > lan-client.log 2>&1 &
CLIENTPID=$!
for i in $(seq 1 180); do "$CURL" -s "http://$API/health" >/dev/null 2>&1 && break; sleep 1; done
echo "client up after ${i}s"
strip < lan-client.log | grep -E "route ready|first hop ping|engine ready" | cut -c1-200

if [ "${QUICK:-0}" = "1" ]; then
    echo "### QUICK: one 256-token completion through the LAN pipeline"
    "$CURL" -s "http://$API/v1/completions" -H "content-type: application/json"         -d "{\"prompt\":\"$(printf 'The quick brown fox jumps over the lazy dog while the patient owl watches from the old oak tree. %.0s' $(seq 1 14))\",\"max_tokens\":32,\"temperature\":0}"         | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['usage']))"
    "$CURL" -s "http://$API/status" | "$PY" -c "import sys,json; d=json.load(sys.stdin); s=d['stats']; print(json.dumps({k:v for k,v in s.items() if k!='stages'})); [print(json.dumps(x)) for x in s['stages']]"
    stop_pids $CLIENTPID $STAGEPID
    echo "### done (quick)"
    exit 0
fi
echo "### bench: local monolithic, local in-process split, remote pipeline"
"$B" --network direct bench --model "$MODEL" --split "1,$N,$LAST" --prompt-tokens "$PROMPT" --gen-tokens "$GEN" --n-gpu-layers -1 --cpu-tensors exps \
    --act-dtype f16 --api-url "http://$API" --log warn 2>&1 | strip | grep -E "^\| |^model:|^layer-range"

echo "### a longer generation through the LAN pipeline (128 tokens)"
"$CURL" -s "http://$API/v1/completions" -H "content-type: application/json" \
    -d '{"prompt":"The history of the Roman Republic begins","max_tokens":128,"temperature":0}' \
    | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['usage'])); print(d['choices'][0]['text'][:400])"

echo "### chat, two turns"
"$CURL" -s "http://$API/v1/chat/completions" -H "content-type: application/json" \
    -d '{"model":"'"$NAME"'","messages":[{"role":"user","content":"In two sentences, why is network latency the main cost when a model is split across homes? /nothink"}],"max_tokens":120,"temperature":0}' \
    | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print(json.dumps(d['usage']))"
"$CURL" -s "http://$API/v1/chat/completions" -H "content-type: application/json" \
    -d '{"model":"'"$NAME"'","messages":[{"role":"user","content":"In two sentences, why is network latency the main cost when a model is split across homes? /nothink"},{"role":"assistant","content":"Every generated token must cross each link in turn."},{"role":"user","content":"How would speculative decoding help? One sentence. /nothink"}],"max_tokens":80,"temperature":0}' \
    | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print(json.dumps(d['usage']))"

echo "### /status"
"$CURL" -s "http://$API/status" | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['prefix_cache'])); s=d['stats']; print(json.dumps({k:v for k,v in s.items() if k!='stages'})); [print(json.dumps(x)) for x in s['stages']]"

stop_pids $CLIENTPID $STAGEPID
echo "### local processes stopped; remote stage left running"
