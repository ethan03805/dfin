# Phase 1 findings

Running notes for design v0.2 phase 1, the encrypted chain. Protocol contract:
[phase1-protocol.md](phase1-protocol.md). Phase 0 results: [phase0-findings.md](phase0-findings.md).

## 1. What was built (6 November 2026)

- **Identities.** Every stage, relay and client has an Ed25519 key. Stages and relays keep
  theirs in a file (`--identity`); clients generate one per start. The public key is the node
  id, 64 hex characters, which a client pins when it dials.
- **Transport.** QUIC with TLS 1.3 (quinn, rustls, ring). Each side presents a self-signed
  certificate carrying its Ed25519 key; the dialer refuses any key but the one it expects;
  the listener records the client's key as the peer id. One request per bidirectional
  stream; results to a return target on unidirectional streams. Idle timeout 30 s, 32
  concurrent streams per connection, frame limits by message type, read timeouts.
- **Sealed route records.** The client seals each hop's instructions (layer range, next hop,
  return target, the next record) to that hop's key with X25519, HKDF-SHA256 and
  ChaCha20-Poly1305. A hop learns its two neighbours and nothing else.
- **Chain forwarding.** Stages dial each other. A forward enters at the first hop and travels
  stage to stage; each hop appends its compute time. Results come back one of three ways:
  reverse (along the chain), direct (the last hop writes to the client's own connection),
  or through a relay.
- **`dfin relay`.** A stream pipe keyed by a client-chosen token. The client reaches the
  first hop through it and the last hop delivers results to it, so no stage sees the
  client's address. The relay sees addresses and ciphertext.
- **Launchers and tests.** `dfin pod` generates a key per stage and probes readiness over
  QUIC (`--relay` adds a relay); `dfin bench --return` and `--relay` select the mode.
  Tests: transport and sealing (8), protocol with one stage and a two-stage chain in
  both modes (2), the daemon end to end in direct, reverse and relay modes (3), plus the
  Phase 0 suites.

## 2. Two things learned the hard way

- **A QUIC reply needs the connection to outlive it.** The relay answered a ping and
  returned from its handler, dropping its last handle to the connection; quinn closed the
  connection and the reply never left. A handler that replies and is done must wait for
  the peer to close (or a short timeout). The stage never hit this because its accept
  loop holds the connection.
- **Benchmarks need an idle machine.** The first real-model run overlapped with a cargo
  build and the Linux image build; the unsplit model decoded at a third of its Phase 0
  speed and the pod numbers were meaningless. The one robust figure from that run is the
  chain's own overhead, round trip minus the stages' compute on the same run: about one
  millisecond in both direct and reverse modes on loopback.

## 3. Real model on one machine

GLM-4.7-Flash Q4_K_M, split 1,24,46 (head 0:1, stages 1:24 and 24:46, tail 46:47),
experts in RAM and attention on the GPU, 256-token prompt, 64 generated, f16 activations.
Log: `phase1-chain-glm47flash-2026-11-06.log`.

| Mode | Prefill | Decode | Output |
|---|---|---|---|
| Unsplit model | 1.6 to 2.5 s | 19.1 to 19.3 tok/s | reference |
| Four pieces in one process | 1.5 s | 19.2 to 19.8 tok/s | identical tokens |
| Pod of four processes, chain, direct return | 28.7 s (cold, unlocked weights) | 16.8 tok/s | identical text |
| Pod of four processes, chain, reverse return | 29.1 s (cold, unlocked weights) | 16.8 tok/s | identical text |

Per decode token on the pod: chain round trip 56.1 ms, of which the two stages
computed 28.1 and 27.3 ms. The chain itself, two link crossings plus the return,
costs 0.7 ms per token in both modes, against 0.8 ms for the Phase 0 client-driven
loop over raw TCP (two round trips at 0.4 ms). Route setup took 5 to 7 ms; a ping to
the first hop 0.05 ms. The 16.8 tok/s is the same figure Phase 0 measured for four
processes sharing this CPU, so the transport change cost nothing. The cold prefill is
the first-request page-fault storm from Phase 0 (the bench pod does not pass
`--mlock`); a serving node locks its weights.

Direct and reverse modes are indistinguishable on loopback because the reverse path
is free there. Over a WAN, reverse adds one link crossing per stage on the way back.

## 4. Two machines

Not yet rerun on the new transport: the MacBook Air still runs the Phase 0 TCP stage. The
follow-up prompt in `mac-node-prompt.md` updates it; `scripts/lan-pod.sh` takes the Mac's
node id and `RETURN=` or `RELAY=`.

## 5. The directory (6 November 2026, later the same day)

`dfin hub`, signed node records, announcements with a reachability check, client-side
route choice from the verified list, a cache for hub outages, and rerouting: the
end-to-end test kills the first stage of a two-stage route after a replacement for its
layers has announced, and the client's next requests succeed over the replacement with
identical output. Protocol section 10 has the details. On the tiny model the whole
sequence (hub up, two stages listed, requests, a third stage listed, the kill, the
reroute) takes under 30 seconds.

## 6. Phase 1 scope still open

Cloaked nodes (stages reachable only through relays), per-layer weight shards with peer
seeding, multi-tensor stage state for `glm5-next` and Kimi K3, failover of a live
session without losing its context, region fencing from several vantage points, fuzzing
of the frame decoders. The Linux build, a CPU-only container image (136 MB)
and continuous integration on GitHub Actions landed in pull request 1 (6.5 minutes
cold; the Windows-generated bindings compile unchanged on Linux x86-64).
