# dfin: Decentralized Frontier Inference Network

dfin runs one large open-weight model across many machines owned by unrelated people.
Each machine holds a contiguous slice of the model's layers and never the whole model.
The first and last layers stay on the user's own machine, so no node sees text or
logits, and every connection goes through Tor, so no node learns who is asking. The aim
is a network that keeps serving frontier open models if hosted endpoints for them
disappear.

**Security and anonymity are the product.** Read what is and is not protected before
relying on it: a node receives intermediate model states and could, with effort,
reconstruct the input from them. Tamper detection, payments and several other required
parts are not built yet.

| To find out | Read |
|---|---|
| What dfin must do, all of it required | [docs/SPEC.md](docs/SPEC.md) |
| What exists today and how it was checked | [STATUS.md](STATUS.md) |
| What gets built next | [docs/PLAN.md](docs/PLAN.md) |
| The bytes on the wire | [docs/protocol.md](docs/protocol.md) |
| How to run it over Tor, and its limits | [docs/anonymity.md](docs/anonymity.md) |
| The rules for changing this repository | [AGENTS.md](AGENTS.md) |

## How a request flows

```
agent framework (OpenCode, Hermes, curl)
        │  OpenAI API, numeric loopback only
        ▼
 dfin client ── head: embedding + layers [0,a)        (user's machine)
        │  hidden states, through Tor to an onion service,
        │  TLS 1.3 inside with the node's key pinned
        ▼
 dfin stage  ── layers [a,b)                           (a stranger's node)
        │  the stage dials the next stage; each knows only its neighbours
        ▼
 dfin stage  ── layers [b,c)                           (another stranger's node)
        │  result back to the client: directly, back along the chain,
        │  or through an optional relay
        ▼
 dfin client ── tail: layers [c,n) + norm + LM head    (user's machine)
                sampling, streaming
```

Every node has an Ed25519 key; its public key is the node id, which dialers pin. Route
instructions are sealed to each hop's key, so a hop learns only its two neighbours. Each
request uses a fresh client key and a fresh route.

Every piece is the same GGUF file opened with a different layer range, so any
architecture llama.cpp supports (including `glm-dsa`, `glm5-next` and `kimi-k3`) works
as a stage, with llama.cpp's quantizations, GPU backends, KV cache types and expert
offload (`--cpu-tensors exps`) intact.

## Get the source

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/ethan03805/dfin.git
cd dfin
scripts/apply-llama-patch.sh        # Windows: .\scripts\apply-llama-patch.ps1
```

`vendor/llama.cpp` is a submodule pinned to an upstream commit; the script applies
`vendor/patches/0001-layer-range.patch` on top. If the shallow submodule fetch fails,
run `git submodule update --init` without `--shallow-submodules`.

## Build

### macOS and Linux

Prerequisites: a C++ toolchain (`xcode-select --install` on macOS; gcc or clang on
Linux), CMake 3.21+, Rust 1.92+ (`rustup`), Python 3.10+ for the test models. On Apple
Silicon the Metal backend is on by default; on Linux pass `GGML_VULKAN=ON` or
`GGML_CUDA=ON` to the script for a GPU.

```bash
scripts/build-llama.sh              # 5 to 15 minutes; TARGETS=llama builds only the library
cargo build --release --locked
```

The Rust build links against `vendor/llama.cpp/build` (override with
`DFIN_LLAMA_BUILD_DIR`) and copies the shared libraries next to the binaries. The
checked-in FFI bindings were generated on Windows x64; regenerate them with
`scripts/gen-bindings.sh` (needs `cargo install bindgen-cli` and libclang) only if the
llama.cpp pin or the patch changes the C API.

### Windows

Prerequisites: Visual Studio 2022 with C++ tools, CMake, Rust 1.92+, Python 3.10+, and
the Vulkan SDK for AMD, Intel or NVIDIA GPUs through Vulkan. Keep the checkout on a
short path, because MSVC breaks beyond 260-character paths. A `subst` drive works:

```powershell
subst W: C:\path\to\this\checkout
cd W:\
.\scripts\build-llama.ps1          # llama.cpp with Vulkan, Release (10 to 20 minutes)
cargo build --release --locked
```

### Docker (Linux, CPU only)

The `Dockerfile` builds llama.cpp and the workspace in a builder stage and ships `dfin`
with its shared libraries on `debian:trixie-slim`. It needs the submodule checked out.

```bash
docker build -t dfin .
docker run --rm dfin --help
```

## Test models and tests

No weights ship with the repository. Tiny random-weight models are enough to exercise
every code path end to end. Generate all of them: tests that cannot find their model
print `SKIP` and pass.

```bash
python3 -m venv .venv
.venv/bin/pip install numpy -e vendor/llama.cpp/gguf-py
.venv/bin/python tools/make_tiny_llama.py models/tiny-llama-f32.gguf
.venv/bin/python tools/shard_gguf.py models/tiny-llama-f32.gguf --out models/tiny-shards
.venv/bin/python tools/make_tiny_kimi_k3.py models/tiny-kimi-k3-res.gguf --attn-res-block-size 2
.venv/bin/python tools/make_tiny_glm5_next.py models/tiny-glm5-next.gguf
cargo test --release --workspace --locked
```

On Windows the interpreter is `.venv\Scripts\python.exe`. The suite needs no Tor: it
uses a local SOCKS fixture. Two slower tests are ignored by default and run in CI: the
real two-minute relay expiry, and full inference through a private nine-node Tor network
(`scripts/test-tor-local.py`, which needs `tor` and `tor-gencert`). See
[docs/anonymity.md](docs/anonymity.md), "Verification".

## Run it over Tor

This is how dfin is meant to be used. Each stage, hub and relay is a Tor onion service
listening on loopback; dfin talks to a Tor daemon you run and never falls back to a
direct connection. [docs/anonymity.md](docs/anonymity.md) has the Tor configuration and
the limits. In outline:

```bash
# on a node: forward an onion service to 127.0.0.1:7001 in torrc, then
dfin stage --model model.gguf --layers 2:6 --identity stage.key --listen 127.0.0.1:7001

# on the user's machine: name the stage by onion address and node id
dfin client --model model.gguf --head 0:2 --tail 6:end \
  --stage '<stage-onion>:7001/<stage-node-id>=2:6' --listen 127.0.0.1:8080
```

Then point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`. `GET /status`
shows the network mode, the route with each stage's node id, timings and return mode,
and states which guarantees are absent.

Instead of naming every stage, run a hub and let stages announce themselves:

```bash
dfin hub    --listen 127.0.0.1:7100 --identity hub.key
dfin stage  ... --hub '<hub-onion>:7100/<hub-node-id>' --advertise '<stage-onion>:7001'
dfin client --model model.gguf --head 0:2 --tail 6:end --hub '<hub-onion>:7100/<hub-node-id>'
```

A stage signs a record of what it serves and announces it; the hub checks the
signature, dials the stage to confirm it answers, and publishes a signed list. The
client verifies both signatures, chooses its own route, keeps a copy of the list for
when the hub is down, and moves to another chain when a stage dies between requests.
The hub never chooses routes for anyone.

### Shards: fetching only your layers

A node that serves a tenth of a 365 GB model should not download all of it.
`tools/shard_gguf.py` cuts a GGUF into one file per layer plus the metadata, embedding
and output head, with a manifest of sizes and SHA-256 hashes. The model id is a hash
over those hashes.

```bash
.venv/bin/python tools/shard_gguf.py model.gguf --out shards/
dfin stage --model shards/model.manifest.json --layers 40:77 --fetch --hub ... --identity node.key
dfin fetch --manifest shards/model.manifest.json --layers 0:3 --hub ...
```

A stage given a manifest and `--fetch` downloads the shards its layers need from nodes
in the directory that serve the same model, checks each against the manifest, and loads
only what it holds.

### Sizing a node

The model file must be byte-identical to the client's (the handshake checks a
fingerprint). Each stage maps only its own layers, so a node needs memory for its range:
roughly `(RAM in GB - 4) / (model GB / layers)` layers. On Apple Silicon use
`--n-gpu-layers -1` and no `--cpu-tensors`; on a PC with a small GPU and a big MoE model
use `--n-gpu-layers -1 --cpu-tensors exps`. Start a stage with `--mlock` so the first
request is as fast as later ones. `--no-mmap` is only safe with memory to spare. Use
`--act-dtype f32` when exact agreement with the unsplit model matters; half precision
is faster and can flip near-tie tokens.

## Development mode: one machine or a LAN

`--network direct` replaces Tor with QUIC between addresses you name. It exposes network
addresses, makes no anonymity claim, and cannot talk to Tor-mode nodes. It exists for
local pods, benchmarks and tests.

```bash
# head = layers 0:2 on the client, one stage 2:5, tail 5:8 on the client
dfin --network direct pod --model models/tiny-llama-f32.gguf --split 2,5 --listen 127.0.0.1:8080

# monolithic versus in-process split versus a local pod
dfin --network direct bench --model models/tiny-llama-f32.gguf --split 3,6 --gen-tokens 64
```

By hand on a LAN, each stage prints its node id at startup and the client names every
stage as `host:port/<node id>=a:b`. Stages are QUIC over UDP, so open the UDP port.

```bash
dfin --network direct stage  --model model.gguf --layers 3:40 --listen 0.0.0.0:7001 --identity node.key --mlock
dfin --network direct client --model model.gguf --head 0:3 --tail 40:end \
  --stage 192.168.1.10:7001/<id>=3:40 --listen 127.0.0.1:8080
```

`scripts/` has the benchmark scripts used for the measurements in STATUS.md; each says
at its top what it needs.

## Fuzzing the decoders

Every byte a node reads from a peer goes through a decoder in `crates/dfin-fuzz`, one
function per wire type, each taking arbitrary bytes and never allowed to panic.
`cargo test` runs them over their seeds, every truncation and bit flip of every seed,
and random noise. For a real fuzzer, `fuzz/` wraps the same functions for libFuzzer
(nightly Rust, Linux or macOS):

```bash
cargo install cargo-fuzz
cargo run --release -p dfin-fuzz --bin dfin-fuzz-seeds -- fuzz/corpus
cargo +nightly fuzz run request -- -max_total_time=120
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request, on
Linux: it builds the patched llama.cpp, audits dependencies, generates the tiny models,
runs the whole test suite, then the two ignored tests (relay expiry and the private Tor
network), and fuzzes every decoder for a short time on nightly.

## License

Apache-2.0 for this repository; see [LICENSE](LICENSE). llama.cpp is MIT.
