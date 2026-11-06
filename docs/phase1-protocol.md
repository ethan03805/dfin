# Phase 1 protocol: identities, transport, chains

Status: being built, 6 November 2026. This is the contract between `dfin-net`,
`dfin-proto` version 2, `dfin stage`, `dfin relay` and the client daemon. Design
background is in [design-v0.2.md](design-v0.2.md) sections 8 and 13.

## 1. Identities

Every node and every client connection has an Ed25519 key.

- A **node key** is long-lived and lives in a file (`--identity path`, created on first
  run, 32 random bytes as hex). Its public key is the **node id**, written as 64 hex
  characters. Reputation and directory records attach to it.
- A **client key** is ephemeral: one per client process start (later: per session).
  It identifies a connection to a stage so that a result can be returned to it, and
  it is what a stage sees instead of an account.
- The same Ed25519 key, converted to X25519, receives sealed route instructions
  (section 4). No separate encryption key is distributed.

## 2. Transport

QUIC (quinn) with TLS 1.3 (rustls). There are no certificate authorities.

- Each side presents a self-signed certificate whose subject public key is its
  Ed25519 key. The dialer knows the node id it expects and refuses any other key.
  The listener accepts any client certificate and records its key as the peer id.
- ALPN `dfin/1`. Idle timeout 30 s with keep-alives every 5 s.
- Every request is one bidirectional stream: the opener writes one frame, the
  acceptor writes one frame in reply, both sides finish. Frames are the Phase 0
  framing (little-endian u32 length, postcard body). Results that travel to a
  return target (section 5) go on unidirectional streams opened by the sender.
- The first stream on a connection to a stage must be `Hello`. Until it succeeds
  the stage answers everything else with `BadRequest`.

Limits, enforced at the receiver: frame size by message type (`Forward` bounded by
`n_batch * n_embd * 4` bytes plus headers, everything else 64 KiB), at most 64 routes
per connection, `n_seq_max` sessions per stage, one `Forward` in flight per session,
10 s to read a control frame, 600 s for a `Forward` reply.

## 3. Messages (dfin-proto version 2)

Requests:

| Message | From | Meaning |
|---|---|---|
| `Hello { version, model_fingerprint, layer_begin, layer_end }` | dialer | Check protocol version, weights and served range |
| `RouteSetup { route, hop }` | previous hop or client | Establish a route; `hop` is this node's sealed instructions (section 4) |
| `Forward { route, session, seq, pos0, act, hops }` | previous hop or client | Run this node's layers; `hops` accumulates per-hop timings |
| `SeqRm { route, session, p0, p1 }` | previous hop or client | Roll the session back; forwarded to the end of the chain, acknowledged in reverse |
| `SessionClose { route, session }` | previous hop or client | Free a session along the chain |
| `RouteClose { route }` | previous hop or client | Tear the route down along the chain |
| `Ping` | anyone | Round-trip measurement |

Replies: `HelloOk { layer_begin, layer_end, n_embd, n_ctx, n_seq_max, backend, node_id }`,
`RouteOk { hops }` (number of hops that acknowledged), `Ack`, `ForwardOk { session, seq,
act, hops }`, `Pong`, `Error { code, message }`.

`route` is a 64-bit number chosen by the client; a node keys routes by (incoming
connection, route), so two clients choosing the same number cannot collide.
`session` and `seq` are as in Phase 0. `hops` is a list of `HopStat { layer_begin,
layer_end, compute_ms }` that each hop appends to before forwarding.

## 4. Route setup

The client knows the whole route: for each hop its address, node id and layer
range. It builds the instructions from the last hop backwards:

```
record_N   = HopRecord { layers, next: None,              ret: <return mode>, inner: [] }
record_i   = HopRecord { layers, next: Some(addr, id_i+1), ret: Reverse,       inner: seal(id_i+1, record_i+1) }
RouteSetup { route, hop: seal(id_1, record_1) }
```

`seal(id, bytes)`: X25519 from the node's Ed25519 key, an ephemeral X25519 key, HKDF-SHA256
over the shared secret with the two public keys as salt, ChaCha20-Poly1305 with a zero
nonce (the key is unique per ephemeral), output `ephemeral_pub || ciphertext`.

A hop unseals its record, checks the layer range is the one it serves, connects to
`next` (pinned to its id) if present, sends `RouteSetup { route, hop: inner }` and waits
for `RouteOk`, then replies `RouteOk` upstream with the count incremented. A hop learns
the address and id of its two neighbours and nothing else: not the client's address
(unless it is hop 1 and no relay is used), not the route beyond its neighbours.

Hop 1 must have received `Hello` on the incoming connection. Later hops check the
model fingerprint and the range from their sealed record instead; the hop-to-hop
connection carries a `Hello` too, with the range `-1:-1` meaning "whatever you serve",
which is how a hop learns its successor runs the same weights before it forwards
anything. Hop-to-hop connections are reused across routes.

## 5. Forwarding and return modes

`Forward` enters at hop 1. Each hop runs its layers, appends its `HopStat`, and sends
`Forward` with its output to the next hop. What happens at the last hop depends on
the return mode in its record:

| Mode | Last hop does | Hop i's reply upstream | Who learns the client's address |
|---|---|---|---|
| `Reverse` | Replies `ForwardOk` on the stream it was asked on | Waits for the downstream reply and relays it | Hop 1 only (or nobody with a relay) |
| `Direct { client_id }` | Opens a unidirectional stream on its connection from `client_id` and writes `ForwardOk` | Replies `Ack` at once, without waiting | Hop 1 and hop N |
| `Relay { addr, node_id, token }` | Connects to the relay, attaches with `token`, writes `ForwardOk` on a unidirectional stream | Replies `Ack` at once | Nobody: the relay sees the client's address and ciphertext only |

In `Direct` mode the client has connected to the last hop (with `Hello`) before
route setup, and the last hop finds that connection by the client's key.

Ordering: the client has at most one `Forward` in flight per session and only sends
the next after receiving the previous result, so forwards for a session arrive in
order at every hop whichever mode is used. `SeqRm`, `SessionClose` and `RouteClose`
are always acknowledged in reverse, so when the client sees `Ack` every hop has done
it.

Errors at hop k in `Direct` or `Relay` mode travel to the return target as an
`Error` frame carrying `route`, `session` and `seq` so the client can match them.

## 6. Relay

`dfin relay` is a QUIC listener with a node key that pipes streams between a client
and a first hop, and between a last hop and the client, keyed by a 16-byte token the
client chose. It never sees plaintext: the client's streams to hop 1 carry sealed
route records and per-link ciphertext terminates at the relay, so the relay sees
frames but every record is sealed to a node key it does not hold, and activations
are sent only by hops, which the relay only sees in the return direction as opaque
bytes. The relay keeps no record of sessions.

Client side: `RelayOpen { token, first_hop: (addr, node_id) }` on the first stream;
afterwards every stream the client opens is piped to a new stream on the relay's
connection to hop 1. Last-hop side: `RelayAttach { token }` on the first stream; every
unidirectional stream it opens is piped to a unidirectional stream on the client's
connection.

Relay is the third return mode, chosen by naming a relay (`--relay host:port/<node id>`).
Without one the client defaults to `Direct`, and `--return reverse` hides its address
from all but the first stage. A relay session that nobody attaches to is forgotten after
two minutes; a relay answers `Ping` so launchers can probe it.

## 7. What a stage keeps

Per incoming connection: the peer id, whether `Hello` passed, the routes it opened.
Per route: the next hop's connection (if any), the return target, the sessions (slot
per session, as in Phase 0). When an incoming connection closes, every route it
opened is torn down along the chain and every session freed. Stages never persist
any of this.

## 8. Status, 6 November 2026

Built and tested: identities, transport, sealed records, route setup, forwarding, all
three return modes, `dfin relay`, the limits in section 2 except fuzzing. Verified with
the tiny model (unit, protocol and end-to-end tests in every mode) and on GLM-4.7-Flash
on one machine (`docs/phase1-findings.md`). Not yet done from the phase 1 list: cloaked
nodes (stages reachable only through relays), the signed directory, per-layer shards,
multi-tensor stage state, replay-log failover, region fencing.

## 9. What changed from Phase 0 and why

- Stages dial each other. A route is a chain, not a star around the client.
- Stages have identities and connections are authenticated, so a route to the wrong
  machine, or a machine that swapped its key, fails before any activation is sent.
- The client's address reaches at most two nodes, and none with a relay.
- Per-hop compute times still reach the client, now as a list in the result.
- Everything a stage accepts from the network has a size and time bound.
