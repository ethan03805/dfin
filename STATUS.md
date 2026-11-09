# Status

What of [docs/SPEC.md](docs/SPEC.md) exists today, what does not, and the test or log
behind each claim. The order in which the missing parts get built is
[docs/PLAN.md](docs/PLAN.md). Last updated 9 November 2026.

Test suite at this revision, on macOS (Apple Silicon): **124 passed, 0 failed, 2 ignored**
with all tiny models generated and nothing skipped. The two ignored tests were run
separately and pass: the two-minute relay expiry, and full inference through a private
nine-node Tor network (`scripts/test-tor-local.py`, Tor 0.4.9.16). CI on Linux has not
yet run this revision. Wire protocol version 3.

## In short

The chain works: a model split into layer ranges, run by separate processes that reach
each other as Tor onion services, with output identical to the unsplit model. That has
been tested on one machine, in CI, and through a private Tor network. It has **not**
been run across separate machines on the public Tor network, and it has not been run at
frontier size.

Most of what makes dfin safe to use with strangers is not built: nothing detects a node
tampering with its slice, nothing stops one node from holding nearly the whole model,
there is no money, and there is one hub. **Until PLAN milestone 3 is done, only run
stages for clients you know, and only list stages on a hub you run.**

## What a user is protected from today

| Property (SPEC) | Today |
|---|---|
| Nodes cannot learn who is asking (8) | Built. Tor only, no fallback, fresh key and route per request. Tested on a private Tor network, not measured on the public one |
| Requests cannot be joined to each other (8.1) | Built for the default. The opt-in conversation session does not exist yet, so every turn prefills its whole conversation |
| No node sees text or logits (6.2) | Built: the client keeps the embedding, first layers, last layer and output head |
| No node holds a whole model (7.1) | **Not enforced.** A client can be pointed at a single stage holding every layer but the client's own |
| Tampering is detected (7.2) | **Not built.** No sanity checks on activations, no redundant execution, no reputation |
| Content is confidential from a node (6) | **No**, and on consumer nodes it will not be: a node can with effort reconstruct its input. The optional attested tier is not built |
| A relay cannot read what it carries (8.2) | **No.** The optional application relay can read model states; `/status` says so |
| Payments cannot be joined to usage (10) | No money exists |
| No single point of failure (12) | One hub. Clients keep routing from a cached directory while it is down |

## Built

| What | Where | How it was checked |
|---|---|---|
| Layer-range patch on mainline llama.cpp: a process loads and runs only its layers and takes or emits the hidden state at the boundary | `vendor/patches/`, `vendor/README.md` | Exact against the whole model in the test suite for tiny `llama`, `kimi-k3` and `glm5-next` models (`crates/dfin-llama/tests`); `glm-dsa` by benchmark script only (`docs/logs/phase0-verify-2026-11-05.log`) |
| Rust bindings and a safe wrapper (model, context, tokenizer, chat templates) | `crates/dfin-llama-sys`, `crates/dfin-llama` | Same tests |
| Stage runtime, client daemon with an OpenAI-compatible API (streaming, sampler), pod launcher, benchmark | `crates/dfin` | End-to-end tests over the tiny model in every return mode (`crates/dfin/tests/e2e.rs`) |
| Tor transport: onion-only destinations through an external Tor daemon, TLS 1.3 with pinned node keys, yamux streams, fresh SOCKS isolation per connection, no direct fallback | `crates/dfin-net/src/stream.rs`, `crates/dfin/src/network.rs`, `docs/anonymity.md` | Transport and privacy regression tests against a local SOCKS fixture; full tiny-model inference through a private nine-node Tor network on Windows, Linux and macOS (`scripts/test-tor-local.py`, run in CI) |
| Request isolation: new client key, route, session and route ids per request; separate keys for directory lookups; routes and caches cleared on success, error and cancellation | `crates/dfin/src/client`, `crates/dfin/src/stage` | `crates/dfin/tests/anonymity.rs`, `crates/dfin/src/stage/privacy_tests.rs`; adversarial review P01 to P16 (`docs/archive/privacy-review.md`) |
| Node identities (Ed25519), sealed per-hop route records, chain forwarding stage to stage, three return modes, `dfin relay` | `crates/dfin-net`, `crates/dfin-proto`, protocol sections 1 to 6 | End-to-end tests in every return mode; relay lifetime tests (`crates/dfin/tests/relay_privacy.rs`) |
| Signed directory: `dfin hub`, node announcements with reachability checks, client-side route choice, a cache for hub outages, rerouting between requests when a stage dies | protocol section 8 | End-to-end test that kills a stage and watches the client move |
| Weight shards: per-layer split files with a hashed manifest, a loader that tolerates absent shards, nodes serving the shards they hold to directory peers | protocol section 10, `tools/shard_gguf.py` | End-to-end test of a stage fetching from a peer; shard path and download hardening tests (`crates/dfin/src/fetch/privacy_tests.rs`) |
| Stage state wider than one residual vector, for `glm5-next` hyper-connections and Kimi K3 attention-residual banks | protocol section 11 | Three-piece pipelines of tiny models match the whole model (`crates/dfin-llama/tests/wide_state.rs`) |
| Hardening: frame size limits by message type, incremental frame reads, read and idle timeouts, route and stream caps, local API origin and host checks | protocol section 2 | Protocol tests (`crates/dfin/tests/stage_proto.rs`), hub and stage privacy tests |
| Fuzzed decoders: one target per wire type, the same targets as ordinary tests, a libFuzzer run in CI | protocol section 13, `crates/dfin-fuzz`, `fuzz/` | CI's first runs found two shape-arithmetic overflows and a tag ambiguity; all fixed with regression tests |
| Development mode: QUIC between named addresses for local pods, benchmarks and tests | `crates/dfin-net/src/quic.rs`, `dfin pod`, `dfin bench` | The end-to-end suite runs on it |
| Linux build, a CPU-only container image, continuous integration | `Dockerfile`, `.github/workflows/ci.yml` | Every pull request builds, audits dependencies, runs the suite, the two slow tests and a short fuzz |

### Measured

All on a LAN in development mode, before Tor was the transport. Nothing has been
measured over Tor.

| Measurement | Result | Log |
|---|---|---|
| GLM-4.7-Flash (30B-A3B, Q4_K_M, 18.5 GB) in four pieces on one machine | Identical text to the unsplit model; 20.6 tok/s against 19.8 | `docs/logs/real-model-glm47flash-2026-11-05.log` |
| The same model across two machines over home WiFi | Identical text; 16.2 to 16.8 tok/s. Measured on the first TCP build, not the current transport | `docs/logs/lan-pod-glm47flash-2026-11-06.log` |
| The same model through the chain on one machine | Identical text at f32; under a millisecond of chain overhead per token | `docs/logs/phase1-chain-glm47flash-2026-11-06.log` |
| The same model served from a partial shard directory | Matches the single file at f32 | `docs/logs/phase1-shards-f32-glm47flash-2026-11-06.log` |

Details and the reasoning behind them are in `docs/archive/phase0-findings.md` and
`docs/archive/phase1-findings.md`.

## Not built

Each line is a milestone of [docs/PLAN.md](docs/PLAN.md), where the tasks are.

| Milestone | Missing |
|---|---|
| 1. Finish the chain | The compartmentalisation cap; opt-in conversation sessions; failover of a live session; encryption across the application relay; locked weights by default; tool-call parsing (`tools` is accepted and ignored); macOS CI, clippy and fmt gates |
| 2. Public Tor network | No run across separate machines over public Tor; no operator runbook tested from a clean machine |
| 3. Tamper detection | Everything: activation sanity checks, redundant execution, shadow admission, reputation, signed logs, trust tiers and pods in the API (a tier suffix on a model name is rejected today) |
| 4. Money | Everything: mint, credits, wallet, per-token payment, payouts, bonds |
| 5. No centre | Several hubs and mirroring, onion seed lists, operator-independence evidence, mint federation, reproducible builds and signed releases, embedded Tor, chaff |
| 6. Frontier scale | GLM-5.3 has never been loaded; the 3 tok/s measurement has not been run; micro-batching, speculative decoding, Kimi K3 boundary rules, FP8 activations, gateway |
| 7. Attested tier (optional add-on) | Everything. The `attested` flag in directory records is always false and Tor-mode hubs and clients reject records that claim it |

## Known problems and limits

- **One test fails intermittently and the cause is unknown.**
  `stage::privacy_tests::repeated_disconnects_release_client_identities_and_engine_handles`
  failed once in CI on Linux (run 42404522275, commit `0e730be`, peer-aborted handshake)
  and once on macOS on 9 November 2026 during a full-suite run ("handshake timed out"
  on a development-mode QUIC connect). It then passed 12 times out of 12 run alone, and
  in two further full-suite runs. PLAN task 1.8.
- **The helper scripts were ported but not all rerun.** `scripts/phase0-verify.sh` was
  run on macOS. `lan-pod.sh`, `real-model-bench.sh`, `real-model-contention.sh` and
  `phase1-shards-check.sh` were made portable (shared helpers in `scripts/common.sh`)
  and syntax-checked only; their Linux and Windows branches have not been run, and
  `lan-pod.sh` gained an unverified `LOCAL_STAGE` setting.
- Hub and relay connections do not check the protocol version; only a stage `Hello` and
  a directory record do.
- **Tests skip silently without their models.** A green run only means something if all
  four tiny model sets were generated first (README, "Test models and tests").
- Half-precision activations between stages flip near-tie tokens on GLM-4.7-Flash; full
  precision reproduces llama.cpp's single-file output exactly. Use `--act-dtype f32`
  when exactness matters.
- The client trusts the one hub it pinned for the whole directory.
- Cold start: a stage should be started with `--mlock`, or the first request pays for
  paging the weights in. PLAN task 1.6 makes that the default.
- The libFuzzer targets need nightly Rust; without it the same targets run as ordinary
  tests, and CI runs the real fuzzer.
- The end-to-end tests spawn processes and load models; on a loaded machine one can time
  out. Rerun that test binary alone before calling it a failure.
- CI covers Linux only. macOS and Windows are built and tested by hand.

## How to check it yourself

```bash
cargo test --release --workspace --locked    # unit, protocol, privacy, fuzz-smoke and end-to-end tests
cargo test --release -p dfin --test e2e      # the process-spawning tests alone
cargo test --release -p dfin-fuzz -- --nocapture   # the fuzz targets over seeds, mutations and noise
cargo +nightly fuzz run request              # a real fuzz run (nightly; see fuzz/Cargo.toml)
python scripts/test-tor-local.py --tor <tor> --tor-gencert <tor-gencert>   # inference through a private Tor network
scripts/phase0-verify.sh                     # benchmark the tiny models three ways (development mode)
scripts/real-model-bench.sh <gguf> <split>   # a real model: whole, split in-process, and as a local pod
```
