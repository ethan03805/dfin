# Strict anonymity mode

dfin now defaults to **Tor-only connections to v3 onion services**. If Tor is unavailable, an onion service is unreachable, or a peer presents the wrong node key, the operation fails. There is no automatic direct-network fallback.

This is a network privacy prototype, not a claim that inference on strangers' hardware is confidential or trustworthy. A node receives intermediate model states and may be able to infer input content or alter results. Attestation, verifiable computation, anonymous credits, and Monero payments are not implemented.

## What happens to a request

1. A local application submits a request to dfin's HTTP API on numeric loopback.
2. The client creates a fresh temporary node key for that request. Directory lookups use separate temporary keys.
3. dfin gives an onion hostname to a trusted local Tor SOCKS proxy. It never resolves that hostname through the operating system. Every connection gets fresh SOCKS isolation credentials.
4. Tor connects to the onion service. Inside that connection, TLS 1.3 authenticates both dfin node keys. The dialer pins the expected server key; a valid onion connection alone is insufficient.
5. yamux carries dfin's request and return streams over the authenticated TCP connection. Existing frame limits and sealed hop records still apply.
6. On success, failure, or cancellation, the client closes the request's routes and clears its prefix cache. The next request starts with another key, new connections, random session/route IDs, and a restarted forward sequence counter.

Once the local HTTP server observes a caller disconnecting, it closes that request's Tor connections, including stalled setup and inference. Directory lookup and retry waits stop, and cancelled queued requests do not contact peers. Cancellation is checked between prompt chunks and decode steps; an active native inference call finishes before that check. Data already sent to a node cannot be recalled. See the [adversarial review](privacy-review.md) for reproductions and test evidence.

The client still runs the embedding and output head locally. A chain can use direct returns, reverse returns, or a dfin relay. Here, **direct return** means a return stream over an existing Tor connection; it does not enable direct IP networking. A dfin relay is optional and is not a replacement for Tor.

The default does not use a dfin relay. When `--relay` is selected, that relay terminates its transport connections and can inspect model states in both directions. Sealed route instructions stay encrypted to the stages; model states are not end-to-end encrypted across the dfin relay. The client warns at startup and reports `relay_can_read_model_states` in `/status`. Tor routing and direct/reverse returns do not require this additional observer. This application relay differs from the legacy UDP cloaking tunnel, which forwards intact encrypted QUIC packets.

## Setup

Install and verify a maintained Tor distribution from the [Tor Project](https://www.torproject.org/download/tor/). Run a dedicated local Tor daemon. dfin does not download, start, or configure Tor automatically.

For a client, a normal Tor configuration includes:

```text
SocksPort 127.0.0.1:9050 IsolateSOCKSAuth
SafeSocks 1
```

Use a private, persistent Tor data directory. Keep Tor's normal public-network defaults and guard state. The fast bootstrap settings in `scripts/test-tor-local.py` belong only to its isolated test network.

For each stage, hub, or relay, add an onion service forwarding to its dfin loopback listener. For example, on a stage's machine:

```text
HiddenServiceDir /absolute/private/path/dfin-stage-onion
HiddenServiceVersion 3
HiddenServicePort 7001 127.0.0.1:7001
```

On Windows, use an absolute private directory in Tor's supported Windows path syntax. Tor writes the service's hostname in that directory. Protect its keys and the dfin node-key file. Never use single-hop/non-anonymous onion-service settings for deployment.

dfin creates new node-key files without replacing an existing path. Unix files are owner-only from creation; on Windows, restrict access to the containing directory because its access rules are inherited. Keep existing keys across restarts. Concurrent creation uses the winner's key or fails closed if its write is incomplete, allowing a later retry.

Start a stage using an existing local model:

```bash
dfin stage --model model.gguf --layers 2:6 \
  --identity stage.key --listen 127.0.0.1:7001
```

The stage prints its dfin node ID. Share both its onion address and node ID through a trusted channel. Configure the client's matching model and layer ranges:

```bash
dfin client --model model.gguf --head 0:2 --tail 6:end \
  --stage '<stage-onion>:7001/<stage-node-id>=2:6' \
  --listen 127.0.0.1:8080
```

Replace placeholders with actual values. For another local Tor port, add `--tor-proxy 127.0.0.1:9150`. This must be a trusted proxy with stream isolation enabled, not an arbitrary SOCKS service. SOCKS authentication cannot prove that a proxy actually uses Tor.

A hub works the same way: forward its onion service to `127.0.0.1:7100`, run `dfin hub`, and give stages `--hub '<hub-onion>:7100/<hub-node-id>' --advertise '<stage-onion>:7001'`. The client can use `--hub` instead of explicit stages. Strict hubs reject records containing direct addresses, hardware/region disclosures, or unsupported attestation claims. The client applies the same restrictions to signed directories and cached records.

## Behavior and limits

| Area | Strict default |
|---|---|
| Outbound networking | Numeric loopback Tor proxy only; destinations must be v3 onion names |
| Incoming service and HTTP listeners | Numeric loopback only; Tor publishes the service |
| Proxy failure or wrong peer identity | Fail the request; never switch to QUIC, DNS, or direct TCP |
| Directory lookups | Separate ephemeral identity from inference |
| Inference requests | New client key and route per request; no prefix reuse across requests |
| Model downloads | Onion peers or offline files; manifest HTTP seeds ignored; explicit HTTP seeds rejected |
| Region claims and geographic probes | Disabled |
| Legacy UDP relay cloaking | Disabled, including remote requests to allocate a UDP port |
| Client and optional hub HTTP APIs | Reject foreign browser origins and non-loopback Host headers; no permissive CORS |
| Unsupported model/trust-tier request | Rejected; never silently treated as an available model |
| Diagnostics | `/status` reports Tor/direct mode, identity rotation, cache policy, and absent content/attestation guarantees |

Local API examples should use `http://127.0.0.1:8080`, not a DNS name. The API does not authenticate local processes. Anyone with access to the machine, its browser profile, logs, model files, or memory may observe activity. Local logs and directory caches are not encrypted.

Tor's local proxy is trusted. Tor cannot guarantee protection against a sufficiently broad traffic observer, compromised endpoints, or colluding participants correlating timing, sizes, repeated prompts, model choice, and route shape. Fresh keys and isolated SOCKS streams remove explicit reuse; they do not guarantee unlinkability or physically disjoint circuits. Stable service identities and operator behavior can link a node's activity. No cover traffic or padding policy is added here.

Request isolation costs extra circuit/connection setup and repeated prefill. Tor latency and TCP head-of-line blocking can be substantial. The first implementation caps a service at 32 accepted connections, each with at most 64 yamux streams and a 64 MiB receive window. TLS accepts have a 10-second deadline; outbound SOCKS plus TLS has a 60-second deadline. These bounds limit resources but do not solve denial of service or Sybil attacks.

## Explicit direct mode

For a LAN experiment or benchmark, every participating command must opt in:

```bash
dfin --network direct pod --model models/tiny-llama-f32.gguf --split 2,6
```

Direct mode uses the existing authenticated QUIC transport and exposes network addresses. `pod`, `bench`, and geographic `probe` require this option. Existing LAN benchmark scripts select it explicitly. Switching modes changes the wire transport; direct and Tor endpoints cannot communicate with one another. Rust callers must likewise choose `Endpoint::direct_client`/`direct_server` or `tor_client`/`tor_server` explicitly.

## Verification

The regular suite runs without Tor:

```bash
cargo test --release --workspace --locked
```

It includes a test-only local SOCKS fixture, large concurrent transport transfers, bidirectional and unsolicited results, wrong-key rejection, isolation credentials, forbidden targets, malicious relay requests, local API protections, signed-directory filtering, and full tiny-model inference in all three return modes. Fixture tests prove application behavior; they provide no network anonymity.

The opt-in real Tor test creates nine Tor processes, four onion services, a hub, two stages, a relay, and clients. All Tor authorities and relays are local; all exits are disabled; public fallback directories are disabled. It checks the same inference flow through real onion circuits, then stops the processes it started. It retains logs/configuration for inspection.

```bash
python scripts/test-tor-local.py --tor /path/to/tor --tor-gencert /path/to/tor-gencert
```

Prerequisites: verified Tor binaries, the patched inference backend, the generated tiny model, Rust 1.92+, and Python 3.9+. A private Tor network validates compatibility, not real-world anonymity, WAN performance, frontier-model scale, or hostile-network resilience. See [the work log](anonymity-work-log.md) for actual results and outstanding checks.

Relay lifetime tests also exercise duplicate tokens, stalled connections, cancellation, return-path ownership, and first-stream deadlines. The real two-minute expiration test is separate from the fast suite and runs explicitly in CI:

```bash
cargo test --release --locked -p dfin --test relay_privacy relay_expires_an_unattached_session -- --ignored
```

## Relationship to Monero

This milestone follows Monero's use of established anonymity networks and separation of identity-bearing activities. It reuses Tor's maintained SOCKS implementation rather than inventing onion routing. The inference protocol needs authenticated interactive streams; Monero's transaction propagation and ledger cryptography do not provide that transport or protect model activations.

Later payment work should use Monero's existing wallet/RPC and address mechanisms. It also needs a separate design for unlinking credit purchases from inference usage, trustworthy compute metering, payout accounting, refunds, and timing/amount correlation. Simply charging in XMR does not make a persistent user balance or usage ledger anonymous. No wallets or real funds are operated by this change.

Primary references: [Monero anonymity networks](https://github.com/monero-project/monero/blob/master/docs/ANONYMITY_NETWORKS.md), [Monero remote-node privacy](https://docs.getmonero.org/interacting/remote-nodes/), [Tor SOCKS extensions and isolation](https://spec.torproject.org/socks-extensions.html), [Chutney](https://gitlab.torproject.org/tpo/core/chutney), [yamux](https://docs.rs/yamux/0.14.1/yamux/).
