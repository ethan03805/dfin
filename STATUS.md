# Status

dfin runs one open-weight model across several machines, each holding a contiguous
range of layers, with the first and last layers on the user's own machine. The point is
a network of strangers' hardware that keeps serving frontier open models if hosted
endpoints for them disappear, with the user unidentifiable by the nodes and the nodes
unable to tamper undetected. The design is `docs/design-v0.2.md`; this page is the one
place that says what of it exists, what does not, and what each missing piece needs.

Last updated 7 November 2026. Phase 0 (one LAN pod) is complete. Phase 1 (the
encrypted chain) is complete except for failover of a live session. Phases 2 (trust and
money) and 3 (scale) have not started.

Strict Tor anonymity is now the default: onion-only destinations, pinned TLS identities,
separate discovery/request keys, and no direct fallback. The Windows workspace suite
passes 71 tests; a separate real Tor integration test also passes on a private
nine-node Tor network. This does not yet
provide content confidentiality, verified computation, or anonymous payment credits.
See [setup and limits](docs/anonymity.md) and [the work log](docs/anonymity-work-log.md).

## Done

| What | Where | How it was checked |
|---|---|---|
| **Phase 0: one LAN pod** | | |
| Layer-range patch on mainline llama.cpp: a process loads and runs only its layers and takes or emits the hidden state at the boundary | `vendor/patches/`, `vendor/README.md` | Exact against the whole model on tiny `llama`, `glm-dsa`, `kimi-k3` and `glm5-next` models, CPU and Vulkan (`crates/dfin-llama/tests`) |
| Rust bindings and a safe wrapper (model, context, tokenizer, chat templates) | `crates/dfin-llama-sys`, `crates/dfin-llama` | Same tests |
| Stage runtime, client daemon with an OpenAI-compatible API (streaming, prefix cache, sampler), pod launcher, benchmark | `crates/dfin` | End-to-end tests over the tiny model in every mode (`crates/dfin/tests/e2e.rs`); `scripts/phase0-verify.sh` |
| A real model on one machine: GLM-4.7-Flash in four pieces, identical output to the whole model, almost no speed loss in-process | `docs/phase0-findings.md`, `docs/logs/real-model-*.log` | `scripts/real-model-bench.sh`, `scripts/real-model-contention.sh` |
| Two machines over WiFi: identical output; the gap to one machine is the WiFi round trip | `docs/phase0-findings.md`, `docs/logs/lan-pod-*.log` | `scripts/lan-pod.sh` with a MacBook Air as the second machine |
| **Phase 1: the encrypted chain** | | |
| Node identities (Ed25519), QUIC with certificates pinned to node keys, sealed per-hop route records, chain forwarding stage to stage, three return modes, `dfin relay` | `crates/dfin-net`, `crates/dfin-proto`, protocol sections 1 to 6 | End-to-end tests in every return mode; GLM-4.7-Flash through the chain with identical text and under a millisecond of chain overhead per token (`docs/phase1-findings.md`) |
| Linux build, a CPU-only container image, continuous integration | `Dockerfile`, `.github/workflows/ci.yml` | Every push builds and runs the whole suite on Linux |
| Signed directory: `dfin hub`, node announcements with reachability checks, client-side route choice, a cache for hub outages, rerouting when a stage dies | protocol section 8 | End-to-end test that kills a stage and watches the client move |
| Cloaked nodes: no inbound port, reachable through a relay-allocated port over the node's own tunnel, end-to-end QUIC, the node never sees a dialer's address | protocol section 9 | End-to-end test through a relay |
| Weight shards: per-layer split files with a hashed manifest, a loader that tolerates absent shards, nodes serving the shards they hold, fetching from directory peers or HTTP seeds | protocol section 10, `tools/shard_gguf.py` | End-to-end test of a stage fetching from a peer; GLM-4.7-Flash from shards matches the single file (f32) |
| Stage state wider than one residual vector, for `glm5-next` hyper-connections and Kimi K3 attention-residual banks | protocol section 11 | Three-piece pipelines of tiny models match the whole model (`crates/dfin-llama/tests/wide_state.rs`) |
| Region fencing: `dfin probe` vantage points, per-probe round trips in the directory, a client fence that excludes cloaked nodes | protocol section 12 | End-to-end test with two probes, one of them far |
| Fuzzed decoders: a libFuzzer target per wire type, the same targets as ordinary tests, generated-value tests for route choice and the hub's checks, a fuzz run in CI | protocol section 13, `crates/dfin-fuzz`, `fuzz/` | CI's first runs found two shape-arithmetic overflows and a tag ambiguity in the first second of fuzzing; all three fixed with regression tests (`docs/phase1-findings.md` section 10) |
| Hardening: frame size limits by message type, read and idle timeouts, route and stream caps per connection | protocol section 2 | Protocol tests (`crates/dfin/tests/stage_proto.rs`) |

## Open

| What | Why it matters | What it needs |
|---|---|---|
| **Phase 1, remaining** | | |
| Failover of a live session: when a stage dies mid-generation the user sees a pause, not a lost conversation | The chain already reroutes between requests; a session in flight still dies with its stage | A per-session replay log on each stage (its emitted hidden states), the client rebuilding the route and replaying into the replacement, an end-to-end test that kills a stage mid-generation. Design v0.1 section 7 |
| Two machines on the new transport | The two-machine result above was measured on the Phase 0 TCP binary; the MacBook still runs it | Someone to run `docs/mac-node-prompt.md` on the Mac, then `scripts/lan-pod.sh` |
| Reproducible builds | Phase 1's exit criterion says a third party reproduces the build; nothing has been done towards it | A pinned toolchain, a build in a container, a published hash, and a second machine that gets the same bytes |
| Phase 1's exit test | "Three pods in three cities serve one model through relays" has not been run | Three machines in three places, which means rented ones |
| **Phase 2: trust and money** (not started) | | |
| Attested stages: a stage image whose measurement the client verifies before trusting it with activations | The only strong statement the network can make about what a node runs; agents are meant to default to attested routes | Server hardware with an enclave (AMD SEV-SNP or Intel TDX), an image build, the verification path in the client, and a disclosure of what attestation does not cover (physical attacks). Design v0.2 sections 4 to 6 |
| Integrity for open nodes: shadow admission, sampled redundancy, signed logs, bonds | Without it a node can return steered activations and only norm checks catch it | A second execution of sampled tokens on another node, a signed log format, and a delisting rule. Design v0.2 section 7 |
| Money: a full-reserve mint, XMR in (USD while a processor allows), per-token payment, daily payouts to operators, reserve proofs | The incentive for strangers to run nodes | A design pass on the mint and wallet, Cashu-style blind credits, and legal review before the mint opens. Design v0.2 section 10 |
| Trust tiers in the API | Users choose pod, attested, ends-only or open routes per request | The tier field, the router honouring it, and the status reporting it |
| **Phase 3: scale** (not started) | | |
| The scale gate: GLM-5.3 at 3 tokens per second or better across 4 to 8 machines on an attested route | The go or no-go for the whole idea at frontier size | Rented machines, sharded loading of a 365 GB model, and the measurements |
| Speculative decoding: MTP head in the client tail, a client-side draft model | Turns a network-bound decode into a usable one | The MTP graph in the patched llama.cpp and the verify path in the client |
| Kimi K3 via anchor pods; hub and mint federation; a measured Tor mode; an agent (OpenCode) completing a real task | The rest of the phase 3 list | Each is its own piece of work once the gate passes |
| **Smaller items, any phase** | | |
| Tool-call parsing in the OpenAI layer | Requests with `tools` are accepted, calls are not parsed | A parser per chat template |
| Micro-batching across sessions on a stage | A stage serving several sessions reads its weights once per step instead of once per session; this is the economic engine of the design | A batching scheduler in the stage engine |
| More than one hub, directory mirroring, fetching the directory through a relay or Tor | One hub is one point of failure and one observer of who asks | Hub peering and a relay path for `DirectoryGet` |
| Probes signing their answers, clients querying probes directly, reputation | The client trusts the hub for everything in the directory today | A signature on each probe answer and a client path to ask probes |

## Known limitations

- Half-precision activations between stages flip near-tie tokens on GLM-4.7-Flash; full precision reproduces llama.cpp's single-file output exactly (`docs/phase1-findings.md` section 7). The scripts use f16 for speed; use `--act-dtype f32` when exactness matters.
- The client trusts the hub it pinned for the whole directory, including probe results. A region fence bounds where a node's network interface is, not where its computation runs.
- Until the integrity checks of phase 2 exist, only run stages for clients you know, and only list stages on a hub you run.
- Cold start is a load-time problem: a stage should start with `--mlock`, or the first request pays for paging the weights in.
- The libFuzzer targets need nightly Rust; on a machine without it the same targets run as ordinary tests, and CI runs the real fuzzer.
- The end-to-end tests spawn processes and load models; on a loaded machine one can time out. Rerun that test binary alone before calling it a failure.

## How to check it yourself

```bash
cargo test                                   # unit, protocol, fuzz-smoke and end-to-end tests (needs the tiny models, see README)
cargo test -p dfin --test e2e                # the process-spawning tests alone
cargo test -p dfin-fuzz -- --nocapture       # the fuzz targets over seeds, mutations and noise, with counts
cargo +nightly fuzz run request              # a real fuzz run (nightly; see fuzz/Cargo.toml)
scripts/phase0-verify.sh                     # the Phase 0 benchmark of the tiny models three ways
scripts/real-model-bench.sh <gguf> <split>   # a real model: whole, split in-process, and as a local pod
scripts/phase1-chain-bench.sh                # a real model through the encrypted chain on one machine
scripts/phase1-shards-check.sh               # a real model served from a partial shard directory
scripts/lan-pod.sh <gguf> <host:port> <N>    # two machines, a stage already running on the other one
```

## Where things are written down

`docs/README.md` lists every document and what it is for.
