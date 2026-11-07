# Strict anonymity work log

This records meaningful changes, decisions, checks, and remaining work. It is not a claim of a security audit.

## Starting point — 7 November 2026

- User confirmed anonymity is the primary value of dfin. Slower inference is acceptable; anonymous mode must be the default and must never fall back to direct connections.
- Waited for the previous agent's protocol/fuzz fixes. GitHub main at `e5d0b1ab13af5cde4560973490fec91c38aa392d` passed both CI jobs in run `42352828257`.
- Created an isolated checkout and branch `codex/strict-anonymity`. Existing checkout, running services, and weights remain untouched. Reused the existing native inference libraries and copied small local test fixtures.
- Baseline Windows release workspace tests passed: 61 tests. Local hardware: Ryzen 5 7600X3D, about 32 GB RAM, Radeon RX 7800 XT. The Mac is not a dependency.

## Transport implementation

- Preserved QUIC as an explicit direct transport. Added a separate Tor TCP transport because Tor does not carry UDP/QUIC.
- Reused the Tor project's SOCKS implementation, rustls for TLS 1.3, and yamux for multiplexing. Existing Ed25519 node-key authentication is retained; no new cryptography is being designed.
- Strict outbound targets must be canonical v3 onion names. Only the numeric loopback Tor proxy is dialled directly. No target DNS lookup or alternative transport is attempted.
- Each connection sends fresh SOCKS authentication credentials for Tor stream isolation. A proxy that selects unauthenticated SOCKS is rejected. The local Tor daemon must be trusted and configured with `IsolateSOCKSAuth`.
- Tor-facing listeners bind only to numeric loopback addresses. TLS handshakes, incoming queues, simultaneous streams, and receive windows are bounded.
- Raised the documented minimum Rust version to 1.92 for the maintained Tor dependency.

## Application policy and request isolation

- Added one network policy shared by stage, client, hub, relay, and fetch. Default: Tor. Direct QUIC requires `--network direct`; Rust callers must also name the direct constructor explicitly.
- Disabled HTTP seed downloads, geographic probes/claims, and legacy UDP relay cloaking in Tor mode. The relay rejects remote requests that would allocate direct UDP ports. Strict listeners and the local API accept numeric loopback addresses only.
- Strict directory records disclose neither hardware nor region, contain onion addresses only, and cannot claim unimplemented attestation. Signed and cached records are filtered before route selection; signatures alone do not make a target safe.
- Separated directory keys from inference keys. Each HTTP generation gets a fresh key and route. Cleared local prefix caches and remote routes after success, error, or cancellation.
- Review found that a daemon-wide forward sequence counter could link fresh identities. Reset it per request and added a hostile-stage regression test checking keys, session IDs, route IDs, and sequence numbers across failed requests.
- Removed permissive browser access in strict mode. Reject foreign Origin and non-loopback Host headers to address cross-site requests and DNS rebinding against the local API.
- Requests for an unknown model or unavailable trust-tier name now fail explicitly. `/status` reports the network mode, isolation/cache policy, and the absence of content confidentiality and verified attestation.
- Updated LAN scripts and the container example to select direct mode explicitly. Updated README, status, protocol notes, and the setup guide. Historical performance measurements retain their original scope.

## Verification on Windows

- The complete release workspace suite passed after the transport, policy, and API changes: 71 tests, with the real Tor test separately opt-in. Added two further regressions (manifest seed leakage and explicit identity continuity); all four application privacy tests then passed. No tiny-model fixture was skipped in these runs.
- Transport tests exchange six concurrent 5 MiB messages, check both return directions, pinned identities, fresh SOCKS credentials, closure behavior, and rejection before even contacting a proxy for non-onion destinations.
- Full application tests compare deterministic tiny-model output with local full coverage, through a hub and two stages in direct-return, reverse-return, and relay-return modes. They also exercise directory announcements, blocked browser requests, absent prefix reuse, and malicious relay requests.
- Acquired Tor Expert Bundle 16.0 from the Tor Project. Checked its published SHA-256 (`2ee123e9167318a3742778ff704e75f01d4780f05a14522d451fe011011077c4`) and valid PGP signature against the official Tor Browser signing-key fingerprint `EF6E286DDA85EA2A4BA7DE684E2C6E8793298290` before execution. Binary: Tor 0.4.9.15, git `4d8efff53a86670b`.
- The current Chutney launcher did not run natively on Windows (it assumes POSIX `SIGKILL`). Added a portable test harness following its private-network topology and settings. Fixed a Windows Tor log-path parsing issue during the first harness attempt.
- **Real Tor test passed:** nine local Tor processes, four onion services, directory discovery, two inference stages, all three return modes, repeated requests, and the local API restrictions. The Rust integration portion completed in 54.06 seconds. This is compatibility evidence, not an anonymity or WAN-performance measurement.
- The harness disabled exits and public fallback directories, used only local authorities and loopback listeners, and stopped every Tor process it started. Existing dfin processes and the original checkout were not targeted.
- Added the real Tor test to Linux CI. Its package installation masks Tor's normal service so only the isolated test network starts. CI results are recorded below when available.

## Design boundaries

- Reused Tor's network transport and existing authentication libraries. Monero's ledger cryptography is not a substitute for inference transport, activation confidentiality, or unlinkable credits.
- No public deployment, paid server, new model download, real wallet, credit mint, or funds movement occurred. The Mac was not needed.
- All inference checks in this milestone used the generated tiny model on CPU. The existing real-model and LAN findings were reviewed, not remeasured here.
- See [anonymity.md](anonymity.md) for setup, maintenance boundaries, and limitations. Outstanding research includes malicious-node integrity, activation inversion, traffic correlation, Sybil/DoS resistance, WAN performance, and unlinkable XMR-backed credits.

## Delivery checks still in progress

Finish the final diff review, publish the review branch, and verify Linux CI including the private Tor test. This log will record the resulting commit and pull request.
