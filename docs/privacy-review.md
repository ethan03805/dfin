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

### P02 — Weak node keys accepted signed records without a secret

**Status:** fixed; all 18 network tests pass. Node IDs accepted low-order Ed25519 points, and directory verification used the library's ordinary verification method. A constructed identity-point key and constant signature passed record verification without any signing secret. Two regression tests failed on the old code: key admission and forged-record rejection.

**Fix:** reject weak keys centrally when converting a node ID to a verifying key, and use the existing library's strict signature verification for records and directories. Normal generated node keys are unchanged. This uses [ed25519-dalek's documented weak-key checks](https://docs.rs/ed25519-dalek/2.2.0/ed25519_dalek/struct.VerifyingKey.html#method.is_weak).

**Scope:** the reproduction forged a record for the attacker's deliberately weak identity. It did not forge a normal existing node's signature or bypass a pinned legitimate TLS key. Rejecting such identities restores the signature-ownership boundary; it does not solve Sybil attacks.

### P03 — Key creation could overwrite identities and exposed a Unix permission window

**Status:** fixed; Windows and Linux checks pass, including Unix permissions and symlinks. Saving a key truncated an existing path, followed symlinks, wrote the secret using default creation permissions, and only then attempted a Unix permission change whose errors were ignored. Concurrent first starts could replace one another's identity; on Unix, default permissions could briefly expose a new secret to other local users.

**Reproduction:** a generated replacement key successfully overwrote an existing identity, failing the new no-overwrite test. The Unix exposure window is established by the write-before-permission-change code order; no cross-user secret capture was attempted.

**Fix:** use exclusive file creation and Unix mode `0600` before writing any bytes, then flush the file. Existing paths are refused by the save operation. If another process wins creation, startup loads that identity; an incomplete or unreadable file fails closed for a later retry. Added collision and Unix symlink/permissions regressions. Windows access control still comes from the containing directory, which must be private.

### P04 — Optional hub HTTP API lacked the client's rebinding protection

**Status:** fixed; all eight application privacy tests pass. The client's API rejected foreign origins and non-loopback Host headers, but a hub started with `--http` did not. Its inspection endpoint exposes the hub's stable node identity and directory. A request with an attacker-controlled Host received HTTP 200 in the regression test.

**Fix:** moved the existing local-origin policy into the shared network module and applied it to both strict-mode HTTP listeners. Native numeric-loopback callers and same-origin browser requests still work. Tests cover both `/health` and `/directory`, foreign/null origins, host lookalikes, and normal local access. Direct-mode behavior is unchanged. The test verifies HTTP admission; a browser exploit was not attempted.

### P05 — Relay reservations and return paths could outlive their owners

**Status:** fixed; five fast relay regressions, eight existing end-to-end tests, and the real two-minute expiration check pass. Three real-process adversarial tests failed on the original relay: a duplicate token replaced an existing session; an outbound TLS connection stalled before insertion bypassed the session limit; and an idle return attachment stayed alive after its original client disconnected.

**Fix:** reserve tokens and capacity atomically before dialing, refuse duplicates, and tie cleanup to a reservation guard. Cancel pending dials when their caller disappears. Permit one return attachment per session, close it when the client disappears, and invalidate the client session when its return path disappears. Bound and own stream-copy tasks so they are cancelled together. Include waiting for the first stream in the first-frame deadline and enforce the documented unattached-session deadline. CI runs the slow expiration check explicitly, keeping ordinary development tests fast.

**Scope:** this closes concrete lifetime and resource-accounting failures; it is not Sybil resistance or a general denial-of-service guarantee. Knowing a relay token remains possession of a return-path capability. Tokens must stay secret and be generated afresh.

### P06 — Historical relay documentation overstates content privacy

**Status:** documentation corrected; client status disclosure tested in all return modes. The relay terminates its client TLS connection and opens a separate TLS connection to the first stage. It copies serialized application frames between them. Sealed route records hide their inner route instructions, but activation frames are not end-to-end encrypted across the dfin relay. Historical protocol text claiming the relay never sees plaintext was inaccurate.

**Change:** corrected the protocol, historical findings, and design boundaries. The client warns when using an application relay and reports `relay_can_read_model_states` in `/status`; normal Tor direct/reverse returns do not require this observer. This is a disclosure correction, not a claim that relay content confidentiality has been implemented. A future fix for that property should reuse an established authenticated end-to-end tunnel, rather than invent a new cryptographic envelope. Legacy UDP cloaking is different and preserves its inner QUIC encryption.

The design also incorrectly implied that client-side route choice prevents a hub from steering. A hub can personalize the directory it returns; the corrected design states this limitation and separates independent-directory work from present guarantees.

### P07 — Directory consumers accepted invalid signed lifetimes

**Status:** fixed; all 88 regular Windows tests pass. Route selection and shard discovery checked expiry and signatures but did not enforce the hub's bounds on lifetimes, future timestamps, text, addresses, or model ranges. A signed future-dated record loaded from a signed cache was accepted by the new regression on the original implementation. A signature authenticates a claim; it does not establish that the claim is valid.

**Fix:** share record validation across the hub, routing, and shard discovery. Reject zero/excessive TTLs, arithmetic overflow, expiry, timestamps beyond the allowed future skew, and invalid field bounds. Clients still accept older records until their bounded TTL expires, preserving outage fallback; newly received hub announcements additionally retain the tighter age check. Directory signatures are now verified before decoding their body because the expected hub key is already known.

**Scope:** a malicious hub can still bias or personalize an otherwise valid directory. This does not add independent directory agreement, Sybil resistance, or proof that a node's self-description is true.

### P08 — Stage route ownership and cancellation were incomplete

**Status:** fixed; all 95 regular Windows tests pass, including seven stage ownership regressions. Five new real-handler tests failed on the original stage: two clients choosing the same route number collided at the next stage; duplicate pending setup was accepted; pending setup bypassed the route cap; cancelled setup kept its downstream connection; and rejected setup retained its downstream connection in an unbounded cache. A sixth regression confirmed that `RouteClose` acknowledged cleanup while an earlier forward still awaited the engine.

**Fix:** each route owns its downstream connection instead of sharing a global connection cache. Route numbers and capacity are reserved before any asynchronous setup. Failed or cancelled setup releases both connections and reservations. A connection owns at most 32 request handlers and cancels/joins them before teardown, preventing work from being installed after cleanup. Explicit route closure cancels operations and waits for their handlers to stop before queuing engine cleanup. Closing routes stay discoverable by disconnect cleanup until that work finishes. Downstream connection closure cascades cancellation without serial network waits. Relay returns are owned by their route; direct returns retain the caller's shared connection.

**Additional check:** reject overflow in an untrusted downstream hop count and release the failed reservation.

**Scope:** established next-hop connections now require a fresh handshake per route, trading some setup time for isolation and bounded ownership. Node identities remain stable public identities. Cancellation cannot interrupt a native CPU/GPU call already executing or recall states already delivered; cleanup follows already queued engine work. These are resource and lifecycle protections, not proof against all denial of service.

### P09 — Hub announcements bypassed capacity and survived cancellation

**Status:** fixed; all 100 regular Windows tests pass, including five hub regressions. Three adversarial tests failed on the original hub: concurrent successful probes stored two records despite a configured limit of one; a duplicate pending announcement launched another probe; and an announcement probe kept running after its caller disconnected.

**Fix:** reserve a record slot and node identity before probing. Count pending new records together with existing entries, allow refreshes at capacity, and cap all simultaneous announcement probes at 32. Reservation guards release capacity after failure or cancellation. Each connection owns at most 32 handlers and cancels/joins them on disconnect. Probe connections close on every exit path, and records are revalidated after probing in case their lifetime elapsed.

**Scope:** these bounds prevent the demonstrated accounting and cleanup failures. They do not stop Sybil identities from competing for capacity or guarantee service during an attack. Configured geographic probe connections remain shared in explicit direct mode; geographic probes are disabled in strict Tor mode.

### P10 — Manifest shard paths could escape the model directory

**Status:** fixed; Windows and Linux checks pass, including the Unix symlink regression. A manifest could name `../private.gguf`, and `serve_shard` followed it outside the model directory. The regression returned a 17-byte dummy private file before the fix; no real identity or personal file was accessed. Filenames and sizes are not bound by the existing model-id calculation, so an unchanged model id alone did not prevent this.

**Fix:** require each shard and `first` to name a single portable GGUF file. Reject path separators, absolute/drive paths, alternate streams, control characters, Windows device names, and invalid suffixes. Require unique split numbers and filenames, a bound first shard, and canonical SHA-256 text. Apply filename validation again at serving/download boundaries, and exclude symbolic links and nonregular files when checking existing shards.

**Scope:** the model directory must remain private to its operator and contain only files intended for sharing. This does not defend against a local process that can change the directory between checking and opening a file, or malicious native-model contents. Model ids continue to identify the shard hash list, not authenticate every manifest field. Temporary-download handling is the next review item.

Reference: [Microsoft's filename and device-name rules](https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file) apply on every platform so moving a manifest to Windows does not change its safety.

### P11 — Downloads clobbered partial files and ignored declared size

**Status:** fixed; all 108 regular Windows tests pass, including five download regressions. The original downloader reused `file.part`, overwriting another attempt's partial file. Peer and HTTP downloads also accepted more bytes than the declared shard size if the larger payload's hash matched. A corrected cancellation regression confirmed that abandoning a download left its partial file behind.

**Fix:** each attempt exclusively creates a randomly named temporary file, held by an ownership guard that removes only that attempt's file on error or cancellation. File creation itself runs in a blocking job whose discarded result still owns cleanup. Peer connections close on cancellation. Refuse peer replies larger than their requested range and HTTP bodies larger than the manifest size before writing them. Check exact final size and SHA-256 before renaming into place; existing partial files are preserved. Unix temporary files are created with mode `0600`; Windows inherits the containing directory's access rules.

**Scope:** this bounds a download by its declared size, not a total disk quota. Abrupt process termination or filesystem errors can still leave randomly named partial files; they are never installed as completed shards without size and hash validation. HTTP seeds remain disabled in strict Tor mode.

Reference: Rust's [`create_new` semantics](https://doc.rust-lang.org/std/fs/struct.OpenOptions.html#method.create_new) provide exclusive creation and refuse an existing symbolic link.

### P12 — Raw destination identities were validated after dialing

**Status:** fixed; all 22 Windows network and transport tests pass. Binary protocol fields construct `NodeId` directly, bypassing its text parser. Two weak destination keys reached the SOCKS proxy before TLS would reject them; the regression observed two proxy contacts where zero were expected.

**Fix:** validate the expected destination key at the shared transport entry point before direct resolution/packets or SOCKS activity. Tests cover both Tor and explicit direct transport. This is early rejection of impossible identity pins, not a TLS authentication bypass or evidence of an IP leak; the existing handshake would still refuse the peer.

### P13 — Unknown API field names were retained and logged

**Status:** fixed; all 112 regular Windows tests pass. The API retained every distinct ignored JSON field name for the process lifetime and printed it in a warning. Field names are arbitrary client-supplied text. A captured-log regression showed a dummy private marker in the log; this test does not claim that ordinary prompt values were logged.

**Fix:** warnings use a small fixed set of recognized unsupported option names and one generic label for all other names. Unknown names are neither copied into permanent warning state nor emitted. A regression checks that 10,000 distinct arbitrary names share one static label, and the captured-log test checks that the private marker is absent. Other operational metadata and error logs still exist as documented.

## Checks completed without a finding so far

- The Tor destination parser runs before opening the proxy socket and rejects IP addresses, ordinary DNS names, onion subdomains, and zero destination ports.
- TLS uses the node key from the certificate and verifies the handshake signature; pinning alone is not substituted for proof of key possession.
- Sealed-route key exchange rejects non-contributory X25519 shared secrets.
- Return-path client identifiers and relay tokens are already included only in the last hop's sealed record; intermediate records use reverse return mode.
- Request sequence counters are already reset and directory lookups already use separate temporary identities.

## Test results

The starting revision passed 73 regular tests, a real Tor integration test on Windows and Linux, and CI wire-decoder fuzzing. New results are recorded here as they complete.

| Change | Check | Result |
|---|---|---|
| P01 reproduction | Hostile stage withholds a forward; caller disconnects | Failed as expected: remote route still open after two seconds |
| P01 fix | Six application privacy tests and eight existing end-to-end tests | 14 passed; real Tor test separately opt-in |
| P01 expanded cancellation coverage | Prefill, setup, and discovery cancellation | 3 passed, 0 failed; no fixture skips |
| P01 published revision `1677bfc` | [Linux CI, private Tor, and fuzzing](https://github.com/ethan03805/dfin/actions/runs/42397113376) | Both jobs passed |
| P02 reproduction | Weak identity and forged-record regressions | Both failed as expected on the original implementation |
| P02 fix | All network unit and transport integration tests | 18 passed, 0 failed |
| P01 + P02 | Complete Windows release workspace | 78 passed, 0 failed; real Tor separately opt-in |
| P03 reproduction | Attempt to overwrite an existing identity | Failed as expected on the original implementation |
| P03 fix | Windows network unit and transport tests | 19 passed before adding concurrent-creator coverage; Unix-specific test awaits Linux CI |
| P03 concurrent creation | Eight simultaneous writers; one persisted identity | Passed, alongside all six Windows identity tests |
| P01–P03 | Complete Windows release workspace | 80 passed, 0 failed; no fixture skips |
| P01–P03 | Real Tor 0.4.9.15, private nine-node network | Passed all return modes; Rust integration portion 50.69 seconds; test processes stopped |
| P04 reproduction | Attacker-controlled Host at the strict hub HTTP API | Failed as expected: received 200 instead of 403 |
| P04 fix | All application privacy regressions, including normal inference | 8 passed, 0 failed; real Tor separately opt-in |
| P05 reproduction | Duplicate token, pending-dial capacity, idle return attachment | All three failed as expected against the original relay |
| P01–P04 revision `73e9f9b` | [Linux CI, real Tor, and fuzzing](https://github.com/ethan03805/dfin/actions/runs/42398169018) | Both jobs passed, including the Unix key-file test |
| P05 fix | Five fast relay tests plus eight existing end-to-end tests | 13 passed, 0 failed |
| P05 expiration | Real two-minute unattached-session deadline | Passed in 120.54 seconds; both client and first-hop connections closed |
| P01–P06 | Complete Windows release workspace | 86 passed, 0 failed; no fixture skips; real Tor and the slow expiration check separately opt-in |
| P01–P06 revision `c1c09fc` | Real Tor 0.4.9.15, private nine-node network | All return modes passed; integration portion 56.65 seconds; children stopped |
| P07 reproduction | Future-dated signed record in an authentic signed cache | Failed as expected: route selection accepted it |
| P01–P07 | Complete Windows release workspace | 88 passed, 0 failed; no fixture skips; two slow integrations separately opt-in |
| P01–P06 revision `c1c09fc` | [Linux CI, real Tor, relay expiration, and fuzzing](https://github.com/ethan03805/dfin/actions/runs/42399468554) | Both jobs passed |
| P08 reproduction | Duplicate/pending route admission, cross-client collision, cancelled/rejected setup | All five failed as expected on the original stage |
| P08 close reproduction | Close a route while its engine reply is withheld | Failed as expected: outstanding operation survived acknowledged closure |
| P01–P08 | Complete Windows release workspace | 95 passed, 0 failed; no fixture skips; two slow integrations separately opt-in |
| P01–P07 revision `f18ce66` | [Linux CI, real Tor, relay expiration, and fuzzing](https://github.com/ethan03805/dfin/actions/runs/42400189246) | Both jobs passed |
| P01–P08 revision `b2408e7` | Real Tor 0.4.9.15, private nine-node network | All return modes passed; integration portion 59.65 seconds; children stopped |
| P09 reproduction | Concurrent capacity, duplicate pending announcement, disconnected caller | All three failed as expected on the original hub |
| P09 targeted fixes | The same three hostile-announcement tests | 3 passed, 0 failed |
| P01–P09 | Complete Windows release workspace | 100 passed, 0 failed; no fixture skips; two slow integrations separately opt-in |
| P10 reproduction | Serve a manifest entry pointing outside the model folder | Failed as expected: returned the dummy private file |
| P01–P08 revision `b2408e7` | [Linux CI, real Tor, relay expiration, and fuzzing](https://github.com/ethan03805/dfin/actions/runs/42400847767) | Both jobs passed |
| P01–P10 | Complete Windows release workspace | 103 passed, 0 failed; no fixture skips; two slow integrations separately opt-in |
| P11 reproduction | Existing partial file, oversized peer response, oversized chunked HTTP body | All three failed as expected on the original downloader |
| P11 cancellation reproduction | Cancel after the temporary file is created | Failed as expected: the file remained |
| P11 targeted fixes | The four download regressions | 4 passed, 0 failed |
| P01–P11 | Complete Windows release workspace, including short/bad-hash download cleanup | 108 passed, 0 failed; no fixture skips; two slow integrations separately opt-in |
| P01–P10 revision `86b3a46` | [Linux CI, real Tor, relay expiration, and fuzzing](https://github.com/ethan03805/dfin/actions/runs/42401459065) | Both jobs passed, including Unix permissions and shard symlink regressions |
| P12 reproduction | Weak raw destination identities with a valid onion target | Failed as expected: two proxy contacts before rejection |
| P12 fix | Network unit and transport tests, including no direct packet / no proxy contact | 22 passed, 0 failed |
| P13 reproduction | Arbitrary ignored request key with a dummy private marker | Failed as expected: marker appeared in captured logs |
| P01–P13 | Complete Windows release workspace | 112 passed, 0 failed; no fixture skips; two slow integrations separately opt-in |
| P01–P11 revision `b3b9c88` | [Linux CI, real Tor, relay expiration, and fuzzing](https://github.com/ethan03805/dfin/actions/runs/42402118564) | Both jobs passed |

Test setup notes: the first P07 test draft referenced an unavailable test helper and overlapped a Windows binary rebuild with the running Tor test, causing a file-lock build failure. The fixture now uses an isolated temporary cache file, and reproduction ran successfully as a separate binary unit test. Neither build failure is counted as a product regression.

The first P11 cancellation fixture waited for buffered bytes to become visible on disk and timed out before exercising cancellation. It now waits for temporary-file creation, and the corrected test failed on the original code because cleanup was absent. Its cleanup assertion permits the file-creation worker to finish after cancellation.
