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

## 6. Cloaked nodes (6 November 2026, evening)

A stage with `--cloak` and no listener runs its QUIC endpoint over a stream to the
relay; the relay binds a public UDP port and shuttles datagrams, substituting synthetic
addresses for the real senders. The end-to-end test puts a cloaked stage behind a
relay, lets the hub probe and list it, and routes a client through it from the
directory alone, with identical output; the whole run takes six seconds on the tiny
model. Two things learned: a reply helper that finishes the stream cannot be used for a
message that opens a tunnel, and quinn's abstract socket interface makes a tunnelled
endpoint about 150 lines, so no second encryption layer was needed. The cost of
cloaking is one extra UDP hop through the relay for every packet to and from that
stage; on a frontier route that is tens of kilobytes per token.

## 7. Weight shards (6 November 2026, late)

The loader patch grew a dozen lines: when a layer range is set, split files absent
from disk are skipped and the tensor count is taken from what is present; a needed
tensor whose shard is absent says so. `tools/shard_gguf.py` writes one split per layer
plus metadata, embedding and output splits, and a manifest whose hashed hashes are the
model id. Pieces loaded from directories holding only their own shards match the full
model to the last bit (`crates/dfin-llama/tests/shards.rs`). Stages serve any shard
they hold; `dfin fetch` and `dfin stage --fetch` pull missing shards from directory
peers or HTTP seeds with hash checks. End to end: a stage started with nothing but the
manifest fetched its two layers and the first shard from a peer found through the hub,
served them, and the client ran head and tail from shards, identical output, in five
seconds on the tiny model. One bug found by the real model: the GGUF writer wants the
byte shape of a quantized tensor, which the reader already provides.

**The real model.** `tools/shard_gguf.py` cut GLM-4.7-Flash Q4_K_M (18.5 GB) into 50
shards (metadata, embedding, 47 layers, output) in a few minutes of disk time. A stage
for layers 24:46 was started from a directory holding only its 23 shards (8.0 GB, 24
files) alongside a stage for 1:24 and a client whose head and tail came from the full
set; it loaded in under 30 seconds with the weights locked, reported the same
resident size as from the single file, and served its layers at the usual speed
(decode-only chain round trip 50.7 ms for stage computes of 26 and 25 ms). Logs:
`phase1-shards-glm47flash-2026-11-06.log` (f16 activations) and
`phase1-shards-f32-glm47flash-2026-11-06.log` (f32).

Three checks say the shard set is byte-faithful: every metadata field compares equal
between the single file and the first shard; sampled tensors of every quantization
type in the file (F32, Q4_K, Q5_K, Q6_K, Q8_0) hash identically; and llama.cpp's own
`llama-completion` produces the same greedy text from the shard set as from the single
file. The pipeline with the partial-directory stage reproduces that text token for
token when activations cross the stage boundaries as f32.

**What this exposed about f16 activations.** With f16 activations between stages,
the same pipeline flips a near-tie token on this prompt after seven tokens
("traditionally dated to 509 BC" becomes "c. 509 BC") and the continuations diverge.
The two-machine WiFi run of 6 November had done the same at its own split point, and
its "longer generation" had never been compared with the single-file text, only the
benchmark prompt had. Phase 0's "identical tokens at f16" therefore holds for the
prompts it tested and not in general: f16 at the boundary is a quality setting, not a
free one, and greedy decoding exposes it on near ties. The exactness tests in this
repository use f32 for that reason; f16 (and later FP8) between stages needs the
perplexity and task measurements the design's open question 3 asked for.

## 8. Wide stage state (6 November 2026, night)

The two architectures Phase 0 could not split now split. The mechanism is a per-model
hook, `stage_state_mult(il)`, that says how many `n_embd` vectors make up the state
entering layer `il`; the loader sets a partial model's input and output row widths from
it, the graph builders unpack a wide input (streams for `glm5-next`, prefix sum plus
bank for Kimi K3) and pack the output the same way, and the activation frame's row
width is whatever the model says. Nothing else in the chain changed; the handshake now
carries both widths so neighbouring pieces check each other.

| Model | Pieces | Widths at the two boundaries | Max logit difference vs whole model |
|---|---|---|---|
| Tiny Kimi K3, residuals every 2 layers, split 2,5 | KDA and MLA layers | 2x and 4x `n_embd` | 2.7e-5 |
| Tiny Kimi K3, split 3,6 | | 3x and 4x `n_embd` | 3.2e-4 |
| Tiny GLM-5-next, 4 hyper-connection streams, split 2,5 | KDA, MLA with k-pool indexer, MoE | 4x and 4x `n_embd` | 9.1e-4 |

The differences are float reassociation from assembling the state in pieces; argmax
agrees at every position and the next decode step agrees as well. Two things had to
be learned on the way: GLM-5-next's attention is nope-only (rope dimension count must be
0, which the generator now does), and a piece with no attention layer builds the
k-pool indexer inputs without consuming them, so their `set_input` must skip
unallocated tensors, the same guard Phase 0 added for the hybrid memory inputs.

## 9. Region fencing (6 November 2026, night)

A hub can now be given vantage points. `dfin probe` is a node of its own: it holds a key,
a region label and a list of hubs allowed to ask it, and on request it pings a node
three times and answers with the best round trip and its label. The hub asks every probe
about each announcing node and publishes the answers in the entry beside its own ping;
cloaked nodes get none, since only their relay could be timed. The client's
`--region us:50` keeps the nodes that at least one probe labelled `us` has measured and
that every such probe reached within 50 ms, picks the route among those, and refuses to
start or reroute when none remain, saying how many the fence excluded. `/status` shows
the fence and marks each hop as fenced or cloaked.

The end-to-end test runs a hub standing in `us`, a `us` probe and an `eu` probe that
adds 100 ms to what it measures (a hidden test flag that stands in for distance), and
one stage: the directory lists three measurements for it, a client fenced to `us:50`
generates the reference text and reports the hop as fenced, and a client fenced to
`eu:50` stops with the count of nodes outside the fence. A unit test covers the rule
itself: a node that one probe in the region reaches in time and another does not is
out, a node with no probe in the region is out, a cloaked node is out even when probed.

What the fence is and is not: it bounds the node's network interface to within about
100 km per millisecond of every probe in the region, which with probes on both US
coasts is a continent-level statement. It does not say where the computation happens,
and it rests on the hub telling the truth about what its probes said, which is already
the trust the client places in the hub it pinned. Letting clients query probes directly,
and probes signing their answers, are the obvious next steps if hubs are ever run by
people other than the probe operators.

## 10. Phase 1 scope still open

Failover of a live session without losing its context, fuzzing of the frame decoders. The Linux build, a CPU-only container image (136 MB)
and continuous integration on GitHub Actions landed in pull request 1 (6.5 minutes
cold; the Windows-generated bindings compile unchanged on Linux x86-64).
