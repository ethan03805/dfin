# Working on dfin

dfin runs one open-weight model across many strangers' machines, each holding a slice
of its layers, so that the user is anonymous to the machines and no machine holds the
whole model. Security and anonymity are the product. Everything else gives way to them.

## Read these, in this order

1. `docs/SPEC.md`: what dfin must do. All of it is required. Section 0 lists the
   owner's decisions; do not reopen them.
2. `STATUS.md`: what exists today and how it was checked. The only place that makes
   claims about the current code.
3. `docs/PLAN.md`: the order of work, each task with a "done when". Pick your task here.
4. `docs/protocol.md`: the wire contract. `docs/anonymity.md`: how Tor mode is set up
   and what it does not protect.

`docs/archive/` is history: earlier designs, findings and review logs. Read it for
background when a document above points there. Where it disagrees with the four
documents above, it is wrong.

## Rules that do not bend

1. **Tor never falls back.** No code path may switch from Tor to a direct connection,
   resolve an onion name through the operating system, or dial anything but the numeric
   loopback Tor proxy in Tor mode. A failure is a failure.
2. **Direct mode is for development.** `--network direct` exists for `pod`, `bench` and
   tests on one machine or a LAN. Do not add features that only work there, and do not
   describe it as a way to use dfin.
3. **Every byte from a peer is hostile.** Bound every frame by message type, grow
   buffers with received data and not with announced lengths, and never panic on input.
   Every wire type has a decoder in `crates/dfin-fuzz` with seeds and a libFuzzer target
   in `fuzz/`; a new wire type is not done without one.
4. **Nothing sensitive in logs or errors.** No prompt or activation content, no
   peer-supplied strings, no route details, no other party's keys or addresses.
5. **No new cryptography.** Reuse Tor, rustls, Ed25519 and the existing sealed boxes;
   for money, the Cashu Development Kit and Monero's own wallet software.
6. **A wire change is a protocol change.** Update `docs/protocol.md`, the protocol
   version, the fuzz seeds and the tests in the same pull request.
7. **Never overclaim.** dfin does not keep content confidential from an open node, and
   splitting a model does not by itself stop a node tampering with its slice (SPEC 6.1
   and 7.1). Do not write or imply otherwise in code, docs, help text or `/status`.
   STATUS.md lists only what a test or a log backs.
8. **No versions.** Do not write "v1", "v2", "phase N", "MVP" or "for now" into docs or
   help text. Work is ordered by PLAN.md and nothing in the spec is optional.
9. **Stop for the owner** before: renting anything, touching real funds or a mainnet
   wallet, deploying a node others can reach, sending model weights to anyone, replacing
   the inference core, or changing SPEC.md beyond fixing an error of fact.

## How work is done

- Branch from `main`; one PLAN task per branch and pull request. `main` is the only
  long-lived branch.
- The pull request that finishes a task ticks it in `docs/PLAN.md` and updates
  `STATUS.md` with what was built and the test or log that shows it.
- Before you call anything done, run:
  ```bash
  cargo build --release --locked
  cargo test --release --workspace --locked
  ```
  CI additionally runs `cargo audit`, the two ignored tests (the two-minute relay expiry
  and the real-Tor integration through `scripts/test-tor-local.py`) and a short fuzz of
  every decoder. If you touched transport, relay or cancellation code, run those too.
- Report results as they are. A test that printed `SKIP` did not pass; say which tests
  you ran and which you only read.
- Dated work logs do not go in `docs/`. Put results in STATUS.md, raw output in
  `docs/logs/` named by what and when, and the reasoning in the pull request.

## Getting a working tree

README.md has the full build instructions for macOS, Linux and Windows. The short form:

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/ethan03805/dfin.git
cd dfin
scripts/apply-llama-patch.sh
scripts/build-llama.sh
cargo build --release --locked
```

Then generate the tiny test models (README, "Test models") and run the tests.

## Traps that have already cost time

- **Tests skip silently without the tiny models.** The end-to-end, anonymity and llama
  tests print `SKIP` and pass if `models/tiny-*.gguf` or the patched llama.cpp is
  missing. Generate all four model sets before trusting a green run.
- **The llama.cpp patch changes.** After pulling, if `vendor/patches/` changed, reset
  the submodule tree, re-apply the patch and rebuild llama.cpp before building Rust.
  `scripts/apply-llama-patch.sh` tells you when the applied patch is stale.
- **Process-spawning tests can time out on a loaded machine.** Rerun that test binary
  alone before calling it a failure, and never run two `cargo` builds in one checkout
  at once.
- **`models/` is not in git** and real weights are large. Never commit weights, node key
  files (`*.key`), directory caches or Tor state.
- **The FFI bindings are checked in** (`crates/dfin-llama-sys/src/bindings.rs`,
  generated on Windows x64). Regenerate with `scripts/gen-bindings.sh` only when the
  llama.cpp pin or the patch changes the C API.

## Where things are

| Path | What |
|---|---|
| `crates/dfin` | The `dfin` binary: `stage`, `client`, `relay`, `hub`, `fetch`, `pod`, `bench` |
| `crates/dfin-net` | Node identities, the Tor/TLS/yamux transport, development-mode QUIC, signed directory records, sealed boxes |
| `crates/dfin-proto` | Wire format: frames, activations, routes, sealed hop records |
| `crates/dfin-llama`, `crates/dfin-llama-sys` | Safe wrapper and raw bindings for the patched llama.cpp |
| `crates/dfin-fuzz`, `fuzz/` | Decoders for every wire type, run as tests and under libFuzzer |
| `vendor/` | llama.cpp submodule and the layer-range patch |
| `tools/` | Tiny test-model generators and the shard tool |
| `scripts/` | Build, bindings, benchmarks, the private-Tor integration test |
| `docs/` | SPEC, PLAN, protocol, anonymity setup, prior art, logs, archive |
