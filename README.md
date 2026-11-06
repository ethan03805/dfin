# Decentralized Frontier Inference Network, Phase 0

Pipeline-parallel inference of large open models across several consumer machines,
with the privacy-relevant pieces (token embedding, first layers, last layer, LM head)
running on the user's own machine. Design: [docs/design-v0.2.md](docs/design-v0.2.md)
(threat model, trust tiers, attested and open nodes, anonymity, money, phases), with the
model physics and execution model in [docs/design-v0.1.md](docs/design-v0.1.md).
Prior art and the current state of llama.cpp: [docs/prior-art-2026-11.md](docs/prior-art-2026-11.md).

Phase 0 is **one LAN pod**: a stage runtime that runs a contiguous layer range of a GGUF,
a client daemon that runs the head and tail and exposes an OpenAI-compatible API, a
local pod launcher and a benchmark. It is also the go or no-go gate from the design:
if a LAN pod cannot hold 3 tokens per second on GLM-5.3 with a cold start under 30
seconds, the WAN version must be rethought before continuing.

## What is here

| Path | What |
|---|---|
| `vendor/llama.cpp` | Mainline llama.cpp (pinned) plus the **layer-range patch**; see `vendor/README.md` |
| `crates/dfin-llama-sys` | Raw bindings to the patched libllama (bindgen, prebuilt DLLs) |
| `crates/dfin-llama` | Safe wrapper: `Model` (full or partial), `Context`, tokenizer, chat templates |
| `crates/dfin-proto` | Wire protocol: postcard frames, f32/f16 activations, model fingerprint |
| `crates/dfin` | The `dfin` binary: `stage`, `client`, `pod`, `bench` |
| `tools/` | Synthetic test-model generators (random weights, valid structure) |
| `scripts/` | Build llama.cpp, regenerate bindings |

### How a request flows

```
agent framework (OpenCode, Hermes, curl)
        │  OpenAI API, localhost
        ▼
 dfin client ── head: embedding + layers [0,a)      (user's machine)
        │  hidden states, f16, TCP (QUIC in phase 1)
        ▼
 dfin stage  ── layers [a,b)                         (pod member)
        ▼
 dfin stage  ── layers [b,c)                         (pod member)
        │
        ▼
 dfin client ── tail: layers [c,n) + norm + LM head  (user's machine)
                sampling, prefix cache, streaming
```

Every piece is the same GGUF file opened with a different layer range, so any
architecture llama.cpp supports (including `glm-dsa`, `glm5-next`, `kimi-k3`) works as
a stage, with llama.cpp's quantizations, GPU backends, KV cache types and expert
offload (`--cpu-tensors exps`) intact.

## Get the source

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/ethan03805/dfin.git
cd dfin
scripts/apply-llama-patch.sh        # Windows: .\scripts\apply-llama-patch.ps1
```

`vendor/llama.cpp` is a submodule pinned to an upstream commit; the script applies
`vendor/patches/0001-layer-range.patch` on top (idempotent). If the shallow submodule
fetch fails, run `git submodule update --init` without `--shallow-submodules`.

## Build (macOS and Linux)

Prerequisites: a C++ toolchain (`xcode-select --install` on macOS; gcc or clang on
Linux), CMake 3.21+, Rust 1.85+ (`rustup`). On Apple Silicon the Metal backend is on
by default; on Linux pass `GGML_VULKAN=ON` or `GGML_CUDA=ON` to the script for a GPU.

```bash
scripts/build-llama.sh              # 5 to 15 minutes
cargo build --release
```

The Rust build links against `vendor/llama.cpp/build` (override with
`DFIN_LLAMA_BUILD_DIR`) and copies the shared libraries next to the binaries. The
checked-in FFI bindings were generated on Windows x64; if `cargo test -p dfin-llama-sys`
fails its layout tests on your platform, regenerate them with `scripts/gen-bindings.sh`
(needs `cargo install bindgen-cli` and libclang).

### Docker (Linux, CPU only)

The `Dockerfile` builds llama.cpp and the workspace in a builder stage and ships `dfin`
with its shared libraries on `debian:trixie-slim` (about 150 MB). It needs the submodule
checked out; the comment at the top says how Vulkan and CUDA variants would be added.

```bash
docker build -t dfin .
docker run --rm dfin --help
```

### Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request: it checks
out the submodule, applies the patch, builds llama.cpp (CPU, `GGML_NATIVE=OFF` so the
cached build is portable between runners), generates the tiny test model and runs
`cargo test --release --workspace` on `ubuntu-latest`. The llama.cpp build tree is cached
on the submodule commit and the patch, so only changes to either rebuild it.

## Build (Windows)

Prerequisites: Visual Studio 2022 with C++ tools, CMake, Rust 1.85+, Python 3.10+,
and the Vulkan SDK for AMD/Intel/NVIDIA GPUs through Vulkan. Keep the checkout on a
short path: MSVC breaks beyond 260-character paths. A `subst` drive works:

```powershell
subst W: C:\path\to\this\checkout
```

```powershell
cd W:\
.\scripts\build-llama.ps1          # llama.cpp with Vulkan, Release (10 to 20 minutes)
cargo build --release
```

### Test model

No weights ship with the repo. A tiny random-weight Llama (1.6 MB) is enough to test
every code path end to end:

```powershell
uv venv .venv; uv pip install --python .venv/Scripts/python.exe numpy -e vendor/llama.cpp/gguf-py
.venv\Scripts\python.exe tools\make_tiny_llama.py models\tiny-llama-f32.gguf
cargo test
```

The integration tests split the tiny model into head, middle and tail pieces and check
that the pipelined logits equal the monolithic ones.

## Run a local pod

```powershell
# head = layers 0:2 on the client, one stage 2:5, tail 5:8 on the client
dfin pod --model models\tiny-llama-f32.gguf --split 2,5 --listen 127.0.0.1:8080
```

Or by hand, one process per machine:

```powershell
dfin stage  --model model.gguf --layers 3:40  --listen 0.0.0.0:7001 --n-gpu-layers -1 --cpu-tensors exps
dfin stage  --model model.gguf --layers 40:77 --listen 0.0.0.0:7001 --n-gpu-layers -1 --cpu-tensors exps
dfin client --model model.gguf --head 0:3 --tail 77:end --stage 192.168.1.10:7001=3:40 --stage 192.168.1.11:7001=40:77 --listen 127.0.0.1:8080
```

On a node that only serves a stage, the model file must be byte-identical to the
client's (the handshake checks a fingerprint); each stage maps only its own layers, so a
node needs memory for its range, not for the whole model. Pick the range by the node's
memory: roughly `(RAM in GB - 4) / (model GB / layers)` layers. On Apple Silicon use
`--n-gpu-layers -1` (Metal, unified memory) and no `--cpu-tensors`; on a PC with a small
GPU and a big MoE model use `--n-gpu-layers -1 --cpu-tensors exps`.

Recommended stage flags on a node that serves one piece: `--mlock` (lock the mapped
weights at load; the first request is then as fast as later ones, at the cost of a longer
start) and the default thread count. `--no-mmap` reads the weights into private memory and
is only safe when the machine has memory to spare beyond the piece.

Then point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`. `GET /status`
shows the route, per-stage round-trip and compute times, and prefix-cache hits.

```powershell
dfin bench --model models\tiny-llama-f32.gguf --split 3,6 --gen-tokens 64
```

prints monolithic versus in-process split versus local-pod-over-TCP decode speed.

## Verification on one machine (5 November 2026)

`scripts/phase0-verify.sh` runs the whole workspace test suite, builds the release binary
and benchmarks every tiny model three ways: monolithic, split in-process (head, middle,
tail as separate pieces in one process) and as a local pod over TCP driven through the
OpenAI API. Log: `docs/phase0-verify-2026-11-05.log`. Ryzen 5 7600X3D, CPU only, f16
activations, 32 prompt tokens, 64 generated, release build:

| Tiny model (8 layers, n_embd 64) | Split | Monolithic ms/token | In-process split | Local pod over TCP | Output |
|---|---|---|---|---|---|
| `llama` | 3,6 | 0.18 | 0.23 | 0.41 (stage RTT 0.41, of which compute 0.34) | identical |
| `glm-dsa` | 2,6 | 0.33 | 0.54 | 0.70 (stage RTT 0.89, compute 0.82) | identical |
| `kimi-k3` | 2,5 | 0.44 | 0.51 | 0.92 (stage RTT 1.02, compute 0.95) | identical |

What this does and does not show: the mechanism is exact end to end for all three
architectures, and the protocol costs about 0.1 to 0.25 ms per hop on localhost (framing,
TCP, f16 conversion). It says nothing about real speeds: a 365 GB model on a LAN pod is
dominated by memory bandwidth and network RTT, which only real hardware can measure.

### Two machines over WiFi

`scripts/lan-pod.sh <model.gguf> <remote host:port> <N> <LAST>` runs the head, a local stage
and the tail here against a stage already running on another machine for layers `1:N`.
Results for GLM-4.7-Flash with a MacBook Air (M4, 16 GB, Metal) serving layers 1:21 and this
PC serving 21:46: 16.2 to 16.8 tok/s decode against 19.8 on one machine, identical text,
prefill of 281 tokens in 1.6 s with the local stage locked in RAM. The gap is the WiFi round
trip (23 ms per token on a link with a 3 ms ping floor); the protocol itself costs 0.4 ms per
hop on loopback. Details in `docs/phase0-findings.md`, log in
`docs/lan-pod-glm47flash-2026-11-06.log`.

### Real model on one machine

`scripts/real-model-bench.sh <model.gguf> <split>` runs the same three modes on a real GGUF,
plus a stock llama.cpp baseline and a live pod with chat requests;
`scripts/real-model-contention.sh` checks how much co-locating several pieces on one CPU costs.
Results for GLM-4.7-Flash (30B-A3B, Q4_K_M, 18.5 GB, experts in RAM and attention on the
GPU) are in `docs/phase0-findings.md` and `docs/real-model-glm47flash-2026-11-05.log`.
Model files live under `models/` and are not part of the repository.

## Phase 0 status

See the task list at the bottom of `docs/design-v0.1.md` section 16 for the exit
criteria. Status of the pieces is tracked in this section as they land.

- [x] Layer-range patch to libllama (loader, graph, KV cache), exact on tiny `llama`, `glm-dsa` and `kimi-k3` models, CPU and Vulkan
- [x] Rust bindings and safe wrapper
- [x] Wire protocol (postcard frames, f32/f16 activations)
- [x] `dfin stage` server (TCP, session slots, automatic rollback on rewind)
- [x] `dfin client` daemon: local head and tail, OpenAI API with streaming, prefix cache, sampler
- [x] `dfin pod` and `dfin bench`
- [x] Verified on tiny synthetic `glm-dsa` and `kimi-k3` models (MLA, indexer, MoE, KDA paths); see `docs/phase0-findings.md` for what each architecture needs at a stage boundary
- [ ] QUIC transport (TCP today)
- [ ] MTP speculative decoding in the client tail
- [ ] Stage state wider than `n_embd` for `glm5-next` (hyper-connections) and Kimi K3 attention residuals
- [ ] Tool-call parsing in the OpenAI layer (requests with `tools` are accepted, calls are not parsed)
- [ ] Micro-batching across sessions on a stage
- [x] Real model on one machine: GLM-4.7-Flash 30B-A3B (deepseek2 arch, MLA + MoE) split into 4 pieces, identical output, 19.5 tok/s in-process versus 19.9 monolithic; see `docs/phase0-findings.md`
- [x] Real model across two machines (PC plus MacBook Air over WiFi): identical output, 16.8 tok/s against 19.8 single-machine, network-bound as predicted
- [ ] The scale gate: GLM-5.3 IQ4_XS across 4 to 8 machines at 3 tok/s or better (moved to phase 3 in design v0.2; needs rented hardware and sharded loading)

What comes next is design v0.2 phase 1, the encrypted chain: transport with node keys in
place of raw TCP, stage-to-stage forwarding with the client touching only the ends through
relays, cloaked nodes, a signed directory, per-layer weight shards, multi-tensor stage state,
a verified Linux build. Nothing in this repository is safe to expose to the internet yet.

## License

Apache-2.0 for this repository. llama.cpp is MIT.
