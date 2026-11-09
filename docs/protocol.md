# dfin wire protocol: identities, transport, chains

This is the contract between `dfin-net`, `dfin-proto` version 3, `dfin stage`,
`dfin relay`, `dfin hub` and the client daemon. Design background is in
[SPEC.md](SPEC.md).

Tor is the default transport; see [strict anonymity](anonymity.md) for its setup and
security boundaries. QUIC and HTTP shard seeds exist only in explicit
`--network direct` mode, which is for LANs, local pods, benchmarks and tests.

Protocol version 3 (9 November 2026) removed UDP relay cloaking and region fencing
from version 2: two requests, two replies, two record fields and the per-probe round
trips in directory entries are gone (sections 9 and 12). Versions do not interoperate.

## 1. Identities

Every node and every client connection has an Ed25519 key.

- A **node key** is long-lived and lives in a file (`--identity path`, created on first
  run, 32 random bytes as hex). Its public key is the **node id**, written as 64 hex
  characters. Reputation and directory records attach to it.
- A **client key** is ephemeral: one per HTTP generation request in Tor mode, with
  separate keys for directory lookups; one per process start in direct mode.
  It identifies a connection to a stage so that a result can be returned to it, and
  it is what a stage sees instead of an account.
- The same Ed25519 key, converted to X25519, receives sealed route instructions
  (section 4). No separate encryption key is distributed.

## 2. Transport

Two transports carry the same application frames, and a process uses exactly one:

- **Tor (default).** TLS 1.3 (rustls) over a SOCKS5 connection through a local Tor
  daemon to a v3 onion service, with yamux multiplexing streams inside TLS. ALPN
  `dfin/1-tor-yamux`. Its stream framing and limits are in section 3.
- **Direct (`--network direct`).** QUIC (quinn) with TLS 1.3 (rustls). ALPN `dfin/1`,
  idle timeout 30 s with keep-alives every 6 s. It exposes IP addresses and provides
  no anonymity.

Tor and direct endpoints cannot talk to each other and neither falls back to the
other. There are no certificate authorities in either. Common to both:

- Each side presents a self-signed certificate whose subject public key is its
  Ed25519 key. The dialer knows the node id it expects and refuses any other key.
  The listener accepts any client certificate and records its key as the peer id.
- Every request is one bidirectional stream: the opener writes one frame, the
  acceptor writes one frame in reply, both sides finish. Frames are the Phase 0
  framing (little-endian u32 length, postcard body). Results that travel to a
  return target (section 5) go on unidirectional streams opened by the sender.
- The first stream on a connection to a stage must be `Hello`. Until it succeeds
  the stage answers everything else with `BadRequest`, except `Ping` and `ShardGet`.

Limits, enforced at the receiver. Frame size by message type: a `Forward` and its
result at most `n_tokens * width * 4` bytes plus 64 KiB of headroom, where a stage
bounds `n_tokens` by its `n_batch` and `width` is the wider of its input and output
rows (section 11); a directory reply at most 16 MiB; a shard reply at most 4 MiB plus
64 KiB; every other frame 64 KiB; nothing above 1 GiB. A reader checks the limit for
the message it expects before reading the body, and grows its buffer as payload
arrives instead of allocating the announced length. Also: at most 64 routes per
connection, `n_seq_max` sessions per stage, one `Forward` in flight per session, 10 s
to read a control frame, 600 s for a `Forward` reply.

Pending setup counts toward the route limit and reserves its route number. Each
route owns a separate next-hop connection; pooling those connections would merge
different clients' route namespaces. Failed/cancelled setup closes its downstream
connections. A stage connection owns at most 32 request handlers and cancels/joins
them before releasing its routes. `RouteClose` cancels outstanding route operations
before queuing engine cleanup. Native inference already executing cannot be interrupted;
cleanup follows it. Disconnects cascade through the route-owned connections.

## 3. Messages (dfin-proto version 3)

### Tor transport framing (default)

The default transport uses TLS 1.3 over a SOCKS5 connection to a v3 onion service,
with mutual node-key authentication and ALPN `dfin/1-tor-yamux`. There is no QUIC/UDP
encapsulation. TLS resumption is disabled in the dialer. Tor SOCKS username/password
authentication is required and every connection gets a fresh random isolation username.

Inside TLS, yamux multiplexes streams. The opener writes one transport byte before
the application payload: `0` for a request/reply stream, `1` for an unsolicited result.
The receiver closes the unused write direction of a result stream. Other marker values
close the connection. Application frames then use the unchanged length/postcard format.
Both sides finish their write direction after a frame; reads remain possible until the
other side finishes. A relay can copy these application streams without changing frames.

Each connection allows at most 64 concurrent yamux streams and a 64 MiB receive window.
Incoming stream queues and concurrent marker readers each have a limit of 32. Marker
reads and inbound TLS handshakes have 10-second deadlines; an outbound SOCKS/TLS attempt
has a 60-second deadline. A listener permits 32 concurrent accepted connections. These
are resource bounds, not guarantees against denial of service. Tor/direct endpoints
cannot negotiate a fallback transport.

### Application messages

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
| `Announce { record }` | node to hub | List me; `record` is a `SignedRecord` (section 8) |
| `DirectoryGet` | anyone to hub | The current `SignedDirectory` |
| `ShardGet { model_id, split, offset, len }` | fetcher to a node | Up to 4 MiB of a shard the node holds; replies `Shard { total, data }` (section 10) |

Replies: `HelloOk { layer_begin, layer_end, n_embd, n_embd_inp, n_embd_out, n_ctx,
n_seq_max, backend, node_id }` (`n_embd_inp` and `n_embd_out` are the widths of the rows
the stage takes and produces, section 11), `RouteOk { hops }` (number of hops that
acknowledged), `Ack`, `ForwardOk { session, seq, act, hops }`, `Pong`,
`Directory { directory }`, `Shard { total, data }`, `Error { code, message }`.

In postcard's encoding the variant index is the first byte. Requests, in order from 0:
`Hello`, `RouteSetup`, `Forward`, `SeqRm`, `SessionClose`, `RouteClose`, `Ping`,
`RelayOpen`, `RelayAttach`, `Announce`, `DirectoryGet`, `ShardGet`. Replies: `HelloOk`,
`RouteOk`, `Ack`, `ForwardOk`, `Pong`, `Directory`, `Shard`, `Error`. A stage sizes its
read from that first byte: only `Forward` (index 2) may be large, and a non-minimal
encoding of the index is refused.

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
for `RouteOk`, then replies `RouteOk` upstream with the count incremented. The sealed
record identifies the immediate successor; only the last hop receives the return
target's identity or relay token. Transport authentication exposes the predecessor's
key, while Tor hides its IP. Direct transport exposes the directly connected peer's IP.
Nodes also observe their model states, message sizes/timing, and forwarded layer/timing
statistics. Sealing instructions does not make route inference or collusion impossible.

Stage errors must not copy downstream addresses, return identities, or peer-provided
error messages upstream. They report the operation and typed error code. Unexpected
responses are diagnosed by type, without dumping their model states or other payloads.

Hop 1 must have received `Hello` on the incoming connection. Later hops check the
model fingerprint and the range from their sealed record instead; the hop-to-hop
connection carries a `Hello` too, with the range `-1:-1` meaning "whatever you serve",
which is how a hop learns its successor runs the same weights before it forwards
anything. Each route owns its hop-to-hop connection; connections are not pooled
across clients or routes.

## 5. Forwarding and return modes

`Forward` enters at hop 1. Each hop runs its layers, appends its `HopStat`, and sends
`Forward` with its output to the next hop. What happens at the last hop depends on
the return mode in its record:

The IP-visibility column below describes explicit direct transport. Tor hides the
client's IP from dfin peers in every return mode; request timing, model state, and
application identities remain visible according to each peer's role.

| Mode | Last hop does | Hop i's reply upstream | Who learns the client's address |
|---|---|---|---|
| `Reverse` | Replies `ForwardOk` on the stream it was asked on | Waits for the downstream reply and relays it | Hop 1 only (or nobody with a relay) |
| `Direct { client_id }` | Opens a unidirectional stream on its connection from `client_id` and writes `ForwardOk` | Replies `Ack` at once, without waiting | Hop 1 and hop N |
| `Relay { relay, token }` | Connects to the relay, attaches with `token`, writes `ForwardOk` on a unidirectional stream | Replies `Ack` at once | No stage sees the client's IP. The relay sees it in direct mode; Tor hides it. The relay can inspect model states |

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

`dfin relay` uses the selected authenticated transport (Tor/TLS by default, explicit
direct QUIC otherwise). It pipes streams between a client and a first hop, and
between a last hop and the client, keyed by a client-chosen 16-byte token.

**Content boundary:** transport encryption terminates at the relay. It can inspect
`Hello`, `Forward`, and return frames, including model states in both directions.
Only the sealed route records remain encrypted to their stage keys. The application
relay is an additional observer of potentially revealing states; it does not provide
end-to-end content confidentiality. This corrects the earlier ciphertext-only claim.
Relay sessions are kept in memory, not persisted.

Client side: `RelayOpen { token, first_hop: (addr, node_id) }` on the first stream;
afterwards every stream the client opens is piped to a new stream on the relay's
connection to hop 1. Last-hop side: `RelayAttach { token }` on the first stream; every
unidirectional stream it opens is piped to a unidirectional stream on the client's
connection.

Relay is the third return mode, chosen by naming a relay (`--relay host:port/<node id>`).
Without one the client defaults to `Direct` returns over the selected transport.
Tor hides the client's IP from all dfin peers in every return mode. In explicit direct
network mode, `--return reverse` exposes the client's IP to the first stage.

A relay reserves a unique token and a session slot before dialing. Pending dials count
towards `--max-sessions`; duplicate tokens and extra return attachments are refused.
Cancelling a caller cancels its dial and stream copies. Losing either the client or
its return path tears down the session. Sessions with no return attachment expire
after two minutes, even if their first-hop connection remains open. The first stream
and frame must arrive within ten seconds. Stream-copy tasks are bounded to 128 for
request/reply directions and 64 for results. A relay answers `Ping` for readiness
checks and any other first request with `BadRequest`.

## 7. What a stage keeps

Per incoming connection: the peer id, whether `Hello` passed, the routes it opened.
Per route: the next hop's connection (if any), the return target, the sessions (slot
per session, as in Phase 0). When an incoming connection closes, every route it
opened is torn down along the chain and every session freed. Stages never persist
any of this.

## 8. Directory

A **node record** is what a stage says about itself: node id, dialable addresses, model
fingerprint and name, layer range, `n_layer`, `n_embd`, KV capacity per session, session
slots, activation type, backend, protocol version, an `attested` flag (always false
today), `issued_at` (unix seconds) and `ttl_secs`. The stage signs the postcard bytes of
the record with its Ed25519 key under the domain `dfin-node-record-v2`, and announces the `SignedRecord` to each hub it was given
(`--hub`, repeatable; `--advertise` names the address others should dial) every
`--announce-every` seconds, with a lifetime of three announcement periods.

The **hub** (`dfin hub`) verifies the signature, checks the record's bounds (protocol
version, one to eight addresses, text lengths, a middle layer range inside the model, a
lifetime of at most an hour, a time stamp within ten minutes of its clock), dials the
first address pinned to the node id and pings it three times, and refuses the
announcement if it does not answer (`Unreachable`). The check is allowed 5 s in direct
mode and 60 s over Tor. The hub keeps the best ping as `hub_rtt_ms`, a routing hint and
nothing more. Records are dropped when they expire. `DirectoryGet` returns a `Directory`
(hub id, `issued_at`, entries sorted by node id, each a signed record plus `hub_rtt_ms`
and `seen_at`) signed by the hub under the domain `dfin-directory-v2`. With `--http`,
the same list is served as JSON for people. In Tor mode the hub lists, and the client
uses, only records whose every address is a v3 onion name, with an empty `backend` and
`attested` false.

Announcements reserve the node identity and directory capacity before the
reachability check; duplicate pending announcements are refused. Refreshes of listed
nodes remain possible at capacity. There are at most 32 simultaneous reachability
checks across the hub and 32 request handlers per connection. Disconnecting cancels
pending checks and releases their reservations. Check connections close on failure as
well as success, and record lifetimes are checked again before insertion.

The hub never assigns routes. A **client** given `--hub host:port/<node id>` and no
`--stage` fetches the directory, verifies the hub's signature against the pinned id and
every record's signature against the node id inside it, drops records for other models,
other protocol versions or that fail the shared bounds/lifetime checks, and chooses a chain that tiles the layers
between its head and tail exactly: fewest hops, then the lowest sum of hub-measured
round trips, then chance. It writes the verified directory to `--directory-cache` and
reads it back when the hub is unreachable. When the route drops, the client fetches the
directory again and prefers a chain that avoids every node of the broken route; if none
exists it retries the old one. `/status` shows the route in use, where the directory
came from, how many nodes it listed and how many were usable.

What the directory does not yet do: more than one hub, mirroring by light nodes,
fetching through an application relay, attestation evidence, and any
reputation.

## 9. Cloaked nodes

Removed 9 November 2026: the UDP cloaking tunnel (`dfin stage --cloak`, `RelayListen`,
`RelayListening`, the record's `cloaked` flag) is gone; onion services replace it.

## 10. Shards

`tools/shard_gguf.py` writes a GGUF as split files llama.cpp loads as one model: the
first holds all metadata and the small global tensors, the second the token embedding,
one per transformer layer, and the last the output head when it is not tied to the
embedding. The manifest (`NAME.manifest.json`) lists each split's number, file, size,
SHA-256, kind and layer, plus optional HTTP seeds. The **model id** is SHA-256 over
the text `dfin-shards-v1`, a newline, and one `no:sha256` line per split, computed
identically by the tool and by `dfin`; it replaces the whole-file fingerprint in
`Hello` and in directory records for sharded models.

A piece needs: the first shard always, the embedding shard if it holds layer 0, the
shards of its layers, and the output shard if it holds the last layer. The llama.cpp
patch lets the loader skip split files that are absent when a layer range is set; a
tensor the range needs whose shard is absent is a clear error.

A manifest's `first` must name split zero. Shard names must be distinct portable
GGUF filenames of at most 200 UTF-8 bytes, with no directories, drive prefixes,
alternate streams, control characters, or Windows device names. Split numbers are
unique and hashes use 64 lowercase hexadecimal digits. Existing shards must be
regular files rather than symbolic links. Keep the model directory private to its
operator and keep private data outside it; the checks do not protect against a
local writer racing file operations. The model id authenticates the hash list,
not every manifest field.

A stage loaded from a shard set answers `ShardGet` for any shard present on its disk
(no `Hello` needed). `dfin fetch` and `dfin stage --fetch` download missing shards:
first from directory nodes that serve the same model id (those whose layer range
contains the shard's layer are tried first, then any), then from seeds. A download
goes to an exclusively created `.dfin-download-<random>.part`, is checked for exact
size and SHA-256, and is renamed into place; a mismatch is deleted and the next
source tried. Oversized peer ranges and HTTP bodies are refused before writing.
Cancellation closes the peer connection and removes that attempt's temporary file;
existing partial files are preserved. Abrupt process termination or filesystem
errors can leave random partial files, but they are not installed as valid shards.
The fetcher keeps retrying until
`--wait` elapses, so a node can start before its peers are listed.

Not yet: swarm-style parallel chunks from several holders, resuming a partial file,
the client fetching its own head and tail, and a signed manifest (the manifest is
trusted as a file today; its hash is what a hub or a seed should publish).

## 11. Stage state wider than one residual vector

Two of the target architectures carry more than the residual stream between layers.
GLM-5.3-Flash (`glm5-next`) keeps four parallel residual streams (manifold
hyper-connections) that every layer mixes; Kimi K3 keeps, besides the running residual,
a bank of raw layer inputs checkpointed every `attn_res_block_size` layers, which every
later layer mixes in and the output head mixes last. A stage boundary inside such a model
must carry all of it.

The wire format needs no change: an `Activations` frame is `n_tokens` rows of
`n_embd` values, and `n_embd` is simply the row width, which for these stages is a
multiple of the model's `n_embd`. What changed is who knows the width. The llama.cpp
patch gives each model class a `stage_state_mult(il)` hook (1 by default; the
hyper-connection count for `glm5-next`; `2 + (il - 1) / block_size` for Kimi K3 with
residuals on), sets the partial model's input and output widths from it, and the two
graph builders split a wide input back into streams or bank plus prefix sum, and
flatten them again at the stage's end. `llama_model_n_embd_inp` and
`llama_model_n_embd_out` report the widths; `dfin` reads them, sends rows of that
width, and `HelloOk` carries `n_embd_inp` and `n_embd_out` so a client checks that the
first hop takes what its head produces, that the last hop produces what its tail takes,
and every hop checks its successor the same way during route setup.

Widths on the wire therefore vary by boundary: for Kimi K3 with block size 8 the state
before layer 93 is 13 vectors, so a late boundary moves thirteen times the bytes of a
plain one, as the Phase 0 findings predicted. The directory record still names the
model's `n_embd`; the per-boundary widths follow from the model and the layer range.

## 12. Region fencing

Removed 9 November 2026: `dfin probe`, `Probe`/`Probed`, region claims, per-probe round
trips and client fences are gone; dfin has no location feature.

## 13. Fuzzing

Every decoder that reads bytes from a peer has a fuzz target in `crates/dfin-fuzz`: a
function from arbitrary bytes to "decoded or not" that must never panic, and that, when
the bytes do decode, does what the real code does next: re-encodes the value and checks
the encoding is stable, converts activations both ways, verifies signatures, opens inner
sealed records, parses addresses. The targets are `request`, `response`,
`return_frame`, `hop_record`, `sealed`, `signed_record`, `signed_directory`,
`frame` (the length-prefixed reader under each limit a receiver uses,
plus the rule that only `Forward` may carry the Forward tag in its first byte),
`node_id_text` and `activations`. Sealed and signed inputs use a fixed key, so a
fuzzer can evolve past the cryptographic checks and reach the code behind them.

The same functions run three ways. `cargo test` runs each over its seeds, every
truncation of every seed, byte and bit flips, random bytes behind a valid prefix, and
pure noise, on every platform, and replays whatever a libFuzzer run left in
`fuzz/corpus` and `fuzz/artifacts`. The `fuzz/` crate at the repository root wraps each
function for cargo-fuzz (nightly; excluded from the workspace), with a seed writer
(`dfin-fuzz-seeds`) that gives libFuzzer valid encodings to start from. CI runs every
target under the address sanitizer for a short time on each push.

The value-level rules of the binary are covered by generated-value tests rather than
byte fuzzing, since the types already decoded: route choice over directories with bad
signatures, empty address lists, absurd layer ranges, NaN round trips, expired and
far-future records, under random exclusions, must never panic and must only
return chains that tile the requested layers with addressable nodes; and the
hub's record check must accept only records with the bounds the hub relies on
afterwards.

Not fuzzed here: QUIC and TLS (quinn and rustls fuzz their own decoders), and the
model computation (llama.cpp's problem; a stage's activation input is bounded in shape
before it reaches the model).

## 14. Status, 9 November 2026

Built and tested: identities, both transports (section 2), sealed records, route setup,
forwarding, all three return modes, `dfin relay`, the signed directory with `dfin hub`
and client-side routing with rerouting when a stage dies (section 8), weight shards
(section 10), wide stage state (section 11), the limits in section 2 and fuzzed decoders
(section 13). Verified with the tiny models (unit, protocol, fuzz and end-to-end tests
in every return mode, including a hub, a stage killed mid-run and a stage fetching its
shards from a peer; the same flow over a private Tor network in CI) and on
GLM-4.7-Flash on one machine in direct mode (`docs/archive/phase1-findings.md`).

Removed on 9 November 2026 with protocol version 3: cloaked nodes (section 9) and
region fencing with `dfin probe` (section 12). Not yet done: replay-log failover of a
live session. The whole picture, done and open, is in `STATUS.md` at the repository
root.

## 15. What changed from Phase 0 and why

- Stages dial each other. A route is a chain, not a star around the client.
- Stages have identities and connections are authenticated, so a route to the wrong
  machine, or a machine that swapped its key, fails before any activation is sent.
- The client's address reaches at most two nodes, and none with a relay.
- Per-hop compute times still reach the client, now as a list in the result.
- Everything a stage accepts from the network has a size and time bound.
