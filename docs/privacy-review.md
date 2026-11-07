# Adversarial privacy review

Started 7 November 2026 from `0512e2d`, the Tor-default implementation in [PR #2](https://github.com/ethan03805/dfin/pull/2). The user authorized continued review, fixes, and testing until stopped. This is an engineering review, not an independent security audit.

## Method and boundaries

Trace every network destination, identity, attacker-controlled message, and request lifetime. Turn suspected failures into reproducible tests before fixing them. Check normal operation after each fix, then exercise real Tor and CI for completed groups of changes. Record unsuccessful hypotheses as well as confirmed findings.

Tests run in the isolated checkout with generated tiny CPU models and loopback-only adversarial peers. Public deployment, paid infrastructure, wallets, and real funds are outside this work. The original checkout remains untouched.

## Review queue

- Request cancellation, blocked peers, queued requests, and cleanup of identity/state.
- Malicious directory records, cached/replayed data, and route selection.
- Stage/relay message handling, resource limits, return paths, and metadata disclosure.
- TLS/node authentication, key storage, and sealed-route boundary checks.
- Local HTTP API origins, request parsing, requested protections, and logs.
- Repeat the full regression suite and private Tor integration after fixes.

## Findings and evidence

### P01 — Cancellation does not interrupt a pending inference request

**Status:** fixed; regression tests pass. The engine noticed a disconnected caller only when sending an output event. It did not check during prompt prefill, and a remote forward could wait up to ten minutes. Cancelled requests could keep remote routes alive and send later prompt chunks.

**Reproduction:** a hostile stage held the first forward indefinitely. Disconnecting the HTTP caller failed the two-second connection-closure assertion on the original implementation. The first test attempt exposed a fixture configuration issue (`prefill_chunk=2` was smaller than four session slots); setting one session made the intended reproduction run.

**Fix:** a request-scoped watcher closes its private endpoint when the HTTP receiver disappears. Discovery and retry delays are cancellable. The engine discards cancelled queued jobs before tokenization/networking and checks between prefill chunks and decode steps. The watcher is aborted when the job finishes. Local cache/route cleanup still runs after success, error, or cancellation.

**Evidence:** all three adversarial cases pass: stalled prefill, stalled route setup, and stalled directory lookup. They require closure within two seconds, no networking from cancelled queued work, and a fresh identity for the next request. The full application privacy suite and all eight existing end-to-end tests also passed before adding the third cancellation case.

**Limit:** already transmitted states cannot be recalled. An active native inference call is not forcibly interrupted; cancellation is checked before the next chunk/step. Prompt cessation depends on the local HTTP server observing the caller's disconnect.

## Checks completed without a finding so far

- The Tor destination parser runs before opening the proxy socket and rejects IP addresses, ordinary DNS names, onion subdomains, and zero destination ports.
- TLS uses the node key from the certificate and verifies the handshake signature; pinning alone is not substituted for proof of key possession.
- Sealed-route key exchange rejects non-contributory X25519 shared secrets.
- Request sequence counters are already reset and directory lookups already use separate temporary identities.

## Test results

The starting revision passed 73 regular tests, a real Tor integration test on Windows and Linux, and CI wire-decoder fuzzing. New results are recorded here as they complete.

| Change | Check | Result |
|---|---|---|
| P01 reproduction | Hostile stage withholds a forward; caller disconnects | Failed as expected: remote route still open after two seconds |
| P01 fix | Six application privacy tests and eight existing end-to-end tests | 14 passed; real Tor test separately opt-in |
| P01 expanded cancellation coverage | Prefill, setup, and discovery cancellation | 3 passed, 0 failed; no fixture skips |
