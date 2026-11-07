#!/usr/bin/env bash
# Real model from shards, one machine: stage A (layers 1:24) from the full shard set,
# stage B (24:46) from a directory holding only its own shards, the client's head and
# tail from the full set; one greedy completion to compare with the single-file run.
#
#   scripts/phase1-shards-check.sh [shard dir] [manifest name]
set -u
cd "$(dirname "$0")/.."
SH="${1:-models/glm-4.7-flash-shards}"
MANNAME="${2:-glm-4.7-flash-q4km.manifest.json}"
MAN="$SH/$MANNAME"
PART=models/glm-4.7-flash-partial-24-46
B=target/release/dfin.exe; [ -f "$B" ] || B=target/release/dfin
PY=.venv/Scripts/python.exe; [ -f "$PY" ] || PY=python3
API=127.0.0.1:${PORT_API:-8080}
PA_PORT=${PORT_A:-7011}; PB_PORT=${PORT_B:-7012}; ACT=${ACT:-f16}
strip() { sed 's/\x1b\[[0-9;]*m//g'; }

echo "### partial shard directory for layers 24:46"
rm -rf "$PART"; mkdir -p "$PART"
"$PY" - "$SH" "$MANNAME" "$PART" <<'PYEOF'
import json, os, shutil, sys
sh, name, part = sys.argv[1:4]
m = json.load(open(os.path.join(sh, name)))
shutil.copy(os.path.join(sh, name), part)
n = 0
for s in m["splits"]:
    if s["kind"] == "meta" or (s["kind"] == "layer" and 24 <= s["layer"] < 46):
        shutil.copy(os.path.join(sh, s["file"]), part)
        n += 1
print(f"copied {n} shards")
PYEOF
echo "$(ls "$PART" | wc -l) files, $(du -sh "$PART" | cut -f1)"

echo "### stage A 1:24 from the full shard set, stage B 24:46 from the partial directory"
"$B" --network direct stage --model "$MAN" --layers 1:24 --listen 127.0.0.1:$PA_PORT --identity shards-a.key \
    --n-gpu-layers -1 --cpu-tensors exps --mlock --n-ctx 8192 --n-seq-max 2 --act-dtype $ACT --log info > shards-a.log 2>&1 &
PA=$!
"$B" --network direct stage --model "$PART/$MANNAME" --layers 24:46 --listen 127.0.0.1:$PB_PORT --identity shards-b.key \
    --n-gpu-layers -1 --cpu-tensors exps --mlock --n-ctx 8192 --n-seq-max 2 --act-dtype $ACT --log info > shards-b.log 2>&1 &
PB=$!
for i in $(seq 1 600); do grep -q "stage ready" shards-a.log 2>/dev/null && grep -q "stage ready" shards-b.log 2>/dev/null && break; sleep 1; done
echo "stages up after ${i}s"
IDA=$(strip < shards-a.log | sed -n 's/.*node id \([0-9a-f]\{64\}\).*/\1/p' | head -1)
IDB=$(strip < shards-b.log | sed -n 's/.*node id \([0-9a-f]\{64\}\).*/\1/p' | head -1)
echo "--- stage B's loader on the partial directory:"
strip < shards-b.log | grep -E "absent|tensors present|resident|skipping it|Error" | cut -c1-160 | head -6

echo "### client: head 0:1 and tail 46:end from the full shard set"
"$B" --network direct client --model "$MAN" --head 0:1 --tail 46:end --stage "127.0.0.1:$PA_PORT/$IDA=1:24" --stage "127.0.0.1:$PB_PORT/$IDB=24:46" \
    --listen $API --n-gpu-layers -1 --cpu-tensors exps --max-sessions 2 --n-ctx 8192 --act-dtype $ACT --log info > shards-client.log 2>&1 &
PC=$!
for i in $(seq 1 300); do curl.exe -s "http://$API/health" >/dev/null 2>&1 && break; sleep 1; done
echo "client up after ${i}s"
strip < shards-client.log | grep -E "route ready|engine ready|Error" | cut -c1-160

echo "### completion (greedy, 64 tokens, activations $ACT)"
curl.exe -s "http://$API/v1/completions" -H "content-type: application/json" \
    -d '{"prompt":"The history of the Roman Republic begins","max_tokens":64,"temperature":0}' > shards-completion.json
"$PY" -c "import sys,json; d=json.load(open(sys.argv[1])); print(json.dumps(d['usage'])); print(d['choices'][0]['text'])" shards-completion.json
curl.exe -s "http://$API/status" > shards-status.json
"$PY" -c "import sys,json; d=json.load(open(sys.argv[1])); s=d['stats']; print(json.dumps(s['chain'])); [print(json.dumps(x)) for x in s['stages']]" shards-status.json

kill $PC $PA $PB 2>/dev/null; sleep 2
echo "### reference from the single-file pipeline (docs/lan-pod-glm47flash-2026-11-06.log):"
sed -n '/^### a longer generation/,/^### chat/p' docs/lan-pod-glm47flash-2026-11-06.log | sed -n 3p | cut -c1-300
rm -rf "$PART" shards-completion.json shards-status.json
echo "### done"
