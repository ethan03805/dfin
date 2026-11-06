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
| `RelayOpen { token, first_hop }`, `RelayAttach { token }` | client, last hop to relay | Section 6 |
| `RelayListen` | cloaked node to relay | Allocate a public port for me; this stream becomes the tunnel (section 11) |
| `Announce { record }` | node to hub | List me; `record` is a `SignedRecord` (section 10) |
| `DirectoryGet` | anyone to hub | The current `SignedDirectory` |

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

## 10. Directory

A **node record** is what a stage says about itself: node id, dialable addresses, model
fingerprint and name, layer range, `n_layer`, `n_embd`, KV capacity per session, session
slots, activation type, backend, protocol version, a free-text region claim, an
`attested` flag (always false in phase 1), `issued_at` (unix seconds) and `ttl_secs`. The
stage signs the postcard bytes of the record with its Ed25519 key under the domain
`dfin-node-record-v1`, and announces the `SignedRecord` to each hub it was given
(`--hub`, repeatable; `--advertise` names the address others should dial) every
`--announce-every` seconds, with a lifetime of three announcement periods.

The **hub** (`dfin hub`) verifies the signature, checks the record's bounds (protocol
version, one to eight addresses, text lengths, a middle layer range inside the model, a
lifetime of at most an hour, a time stamp within ten minutes of its clock), dials the
first address pinned to the node id and pings it three times, and refuses the
announcement if it does not answer (`Unreachable`). It keeps the best ping as
`hub_rtt_ms`, the first evidence for region fencing. Records are dropped when they
expire. `DirectoryGet` returns a `Directory` (hub id, `issued_at`, entries sorted by node
id, each a signed record plus the hub's observations) signed by the hub under the domain
`dfin-directory-v1`. With `--http`, the same list is served as JSON for people.

The hub never assigns routes. A **client** given `--hub host:port/<node id>` and no
`--stage` fetches the directory, verifies the hub's signature against the pinned id and
every record's signature against the node id inside it, drops records for other models,
other protocol versions or that have expired, and chooses a chain that tiles the layers
between its head and tail exactly: fewest hops, then the lowest sum of hub-measured
round trips, then chance. It writes the verified directory to `--directory-cache` and
reads it back when the hub is unreachable. When the route drops, the client fetches the
directory again and prefers a chain that avoids every node of the broken route; if none
exists it retries the old one. `/status` shows the route in use, where the directory
came from, how many nodes it listed and how many were usable.

What the directory does not yet do: more than one hub, mirroring by light nodes,
fetching the directory through a relay or Tor, attestation evidence, and any
reputation.

## 11. Cloaked nodes

A stage started with `--cloak host:port/<relay id>` and no `--listen` opens no port.
It dials the relay, opens one bidirectional stream, and sends `RelayListen`. The relay
binds a UDP port on its public address, answers `RelayListening { addr }` on that
stream without finishing it, and from then on the stream is a tunnel: every datagram
that reaches the port goes down the stream as a frame (`u16` length, 16-byte
IPv6-mapped address, `u16` port, payload), and every frame the node sends up goes out
of the port to the address named in it.

The address the relay attaches is **synthetic**: one from 100.64.0.0/10 allotted per
real sender, mapped back only inside the relay. The node's QUIC endpoint runs over the
tunnel and sees those synthetic addresses as its peers, so a cloaked node learns no
address of any dialer, neighbour, hub or client, while dialers learn only the relay's
address and port.

Dialers need nothing new: the directory record lists `relay-host:port` as the node's
address, with `cloaked: true`, and a dial to it is an ordinary QUIC connection pinned to
the node's key. The handshake and every byte after it are end to end between dialer
and node; the relay forwards ciphertext it cannot read. The hub's reachability probe
and `hub_rtt_ms` therefore describe the path to the relay, and the client treats
`cloaked` records as unplaceable for any region preference.

A cloaked node's stage still dials out normally (to the hub, to its next hop, to the
relay) from an ephemeral port. `--cloak` is repeatable; each relay yields one public
address and the record lists them all. When the tunnel breaks the stage re-registers
after five seconds, usually getting a new port, and its next announcement carries it.
The relay refuses registrations beyond `--max-cloaked`, allocates from `--port-range`
when one is given, and needs a public host (`--public-host`, or a specific `--listen`
address) to name the ports by.

## 8. Status, 6 November 2026

Built and tested: identities, transport, sealed records, route setup, forwarding, all
three return modes, `dfin relay`, the signed directory with `dfin hub` and client-side
routing with rerouting when a stage dies (section 10), cloaked nodes (section 11), the
limits in section 2 except fuzzing. Verified with the tiny model (unit, protocol and
end-to-end tests in every mode, including a hub, a stage killed mid-run, and a cloaked
stage behind a relay) and on GLM-4.7-Flash on one machine (`docs/phase1-findings.md`).
Not yet done from the phase 1 list: per-layer shards, multi-tensor stage state,
replay-log failover of a live session, region fencing from several vantage points.

## 9. What changed from Phase 0 and why

- Stages dial each other. A route is a chain, not a star around the client.
- Stages have identities and connections are authenticated, so a route to the wrong
  machine, or a machine that swapped its key, fails before any activation is sent.
- The client's address reaches at most two nodes, and none with a relay.
- Per-hop compute times still reach the client, now as a list in the result.
- Everything a stage accepts from the network has a size and time bound.
