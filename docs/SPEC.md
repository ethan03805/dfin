# dfin: what it must do

This is the specification of dfin, the Decentralized Frontier Inference Network. **Everything in it is required.** There are no versions of dfin and no optional subset: the order the work happens in is [PLAN.md](PLAN.md), what exists today is [STATUS.md](../STATUS.md), and the bytes on the wire are [protocol.md](protocol.md). Where an older document disagrees with this one, this one wins.

It describes the intended system, including protections that are not built yet. Never read a sentence here as a claim about the current code; STATUS.md is the only place that says what exists.

Background this spec relies on but does not repeat, in [archive/design-v0.1.md](archive/design-v0.1.md): model sizes and the physics budget (sections 2 and 3), the execution model (6), replay logs and failover (7), routing (8), weight distribution (13) and the API (14). It is cited below as "v0.1 section N". Measurements are in [archive/phase0-findings.md](archive/phase0-findings.md) and [archive/phase1-findings.md](archive/phase1-findings.md).

A network that runs frontier-scale open-weight models, Chinese-lab models first (GLM-5.3, then Kimi K3), on machines owned by many unrelated people, and exposes them through an OpenAI-compatible endpoint. It exists so that these models stay reachable from outside the labs' own endpoints and outside any single cloud, however unpopular hosting them becomes. Users are anonymous to the machines that serve them. No node ever holds a whole model. Tampering by a node is detected, and what a node could learn about content is told to the user in plain words. Operators are paid, in Monero, in proportion to the work that runs through their machines.

---

## 0. Decisions that shape this spec

Settled by the owner on 9 November 2026. An agent does not reopen these; it raises a conflict with the owner instead.

| Decision | What it means |
|---|---|
| **Security and anonymity are the product** | When speed, convenience or scope conflict with them, they win. Splitting a model and running it is treated as the solved part |
| **No versions** | Everything here is required. Work is ordered by PLAN.md, never labelled v1, v2 or "later release" |
| **Tor only** | Every node is a Tor onion service and every connection goes through Tor. A failure never falls back to a direct connection |
| **Direct mode is a development tool** | `--network direct` (QUIC on a LAN) exists for local pods, benchmarks and tests. It carries no anonymity claim and is not a product mode |
| **No location feature** | Region fencing, timing probes and region claims are removed. An onion service has no location, and that is the point |
| **No UDP cloaking** | The relay-allocated-port tunnel is removed. Being an onion service is how a node hides its address |
| **Compartmentalisation** | No node may hold or run a whole model. The client keeps the first and last layers, and the rest is spread over several independent nodes (section 7.1) |
| **The consumer network stands on its own** | Anonymity and tamper detection must hold on consumer hardware with no enclave anywhere. Enclave-attested nodes are an optional add-on built last (sections 5 and 6.3), never a dependency and never the default |
| **Fresh identity per request, sessions by opt-in** | By default every request uses a new key, new route and no cache. A user may opt a conversation into one identity to get prefix caching and failover, and is told what that links (section 8.1) |
| **Money is Monero** | Credits bought and redeemed in XMR, blind-signed so purchases and usage cannot be joined (section 10) |

Earlier decisions that still hold: the adversary is deplatforming, not only a curious node; strangers run nodes for pay from the start; chain forwarding (stage hands to stage) is the pipeline shape; Monero is the reference for keys-not-accounts, money and release practice; a stage's state is as wide as the architecture needs.

---

## 1. Goals, non-goals and constraints

**Goals**

1. Serve GLM-5.3 and, later, Kimi K3 through an OpenAI-compatible `/v1/chat/completions` endpoint with streaming and tool calling, from machines that are neither the model labs' own endpoints nor any single cloud.
2. Anyone can run a node and earn. Anyone can use the network either by contributing compute or by paying, in Monero, without an account.
3. No node, relay, hub, mint or payment links a request to a person or an address.
4. A node cannot alter the model's behaviour without being detected, on consumer hardware, with no enclave involved. What a node could learn about a user's content is disclosed, and the user always knows which tier served them.
5. No node holds or runs a whole model.
6. No single party whose removal stops the network, including the operator of the first directory and mint.
7. Node churn mid-generation produces a pause, not an error, for a conversation that opted into a session.

**Not goals**

- Hosted-API latency. Section 13 gives the measured and projected numbers.
- Cryptographic confidentiality of content on consumer nodes. Section 6 explains why it is not available and what the network does instead.
- Any statement about where a node is. Section 9.
- Guaranteed resistance to a global passive adversary. Everything goes through Tor, but timing and volume correlation remain threats.
- Vision input, fine-tuning, training.

**Constraints** are those of v0.1 section 1 (consumer latency, upload bandwidth, memory bandwidth, model licences), plus what the first LAN measurements showed: the protocol itself costs about 0.4 ms per hop; a home WiFi link costs 15 to 25 ms per token; a stage must lock its weights at load or its first request takes ten to thirty seconds.

---

## 2. Threat model

| Adversary | Wants | Can do | What the design does |
|---|---|---|---|
| **Deplatformer**: a cloud, a payment processor, a model host, a code host, a registrar, an app store | The service gone | Refuse hosting and payments, delist weights, take down repositories and domains, pressure identifiable operators | No cloud dependency. Nodes on many small providers and in homes. Weights seeded peer to peer under signed manifests. Software mirrored, reproducibly built, signed by several keys, discoverable without DNS (seed lists in the binary, onion addresses). Directories and mints federated. The XMR rail cannot be refused. Every node is an onion service, so an operator's address is never published |
| **Open node**, curious or malicious | Prompts, identities, steered outputs, pay without work | Invert the activations it receives (section 6.1), return wrong or steered activations, return garbage, claim work it did not do | Client holds embedding, first layers, last layer and output head, so no node sees text or logits. No node holds the whole model. Tor keeps addresses away. Chaff sessions. Redundant execution and shadow admission catch tampering. Payment per token on delivery. Bonds. The user is told in plain words what an open node can do |
| **Attested node operator** with physical access | The same, through the enclave | Memory-bus interposer attacks have broken SGX and SEV-SNP on DDR4 and reportedly reach DDR5 and TDX; vendors call physical attacks out of scope | Attestation still proves the code. Residual physical risk is disclosed. Platforms whose attestation carries cloud-provider evidence rank above self-hosted enclaves. Redundancy sampling continues at a low rate |
| **Relay operator** | Client addresses and content | A dfin application relay sees session tokens, neighbouring onion addresses, and model-state frames | Tor hides the client IP from the relay. The default uses no dfin relay. Activations must be encrypted end to end across an application relay; sealed route records alone do not provide it |
| **Hub or mint operator** (the project, at first) | Everything a central party could want | Bias or personalize a directory, censor nodes, steal reserves, link a purchase to a buyer | Clients choose routes, but a hub can influence the available choices, so clients compare independent directories. Mints are full-reserve with public proofs and issue blind credits |
| **Network observer** (ISP, backbone) | Who talks to whom | Timing correlation | Every link encrypted and carried by Tor. Not defended against globally |
| **Malicious client** | Free service, denial of service, probing nodes | Flood a stage, not pay, send malformed frames | Prepaid per token. Session and frame limits. Rate limits per connection. Parsers fuzzed |
| **Sybil operator** | Many fake nodes to grab routes, harvest activations, farm credits, or quietly hold every piece of a model | Register thousands of keys | Shadow admission with verified work. Bonds. Reputation that takes time to earn. Routes prefer long-lived keys. Chaff makes harvested sessions ambiguous. Compartmentalisation (section 7.1) is only as strong as the evidence that a route's nodes belong to different operators, which is an open problem (section 17) |
| **Model licensors** | Their terms honoured | Legal action over redistribution | Counsel reviews both licences before weights are shipped to anyone outside the core team (section 17) |

**What never exists in one place.** No participant holds both a client IP and request content. No participant holds both a route and content. No participant holds both a purchase identity and a usage record. The one deliberate exception is the gateway (section 3), which exists for users who will not run the daemon and is therefore a trusted or attested party by definition.

---

## 3. Participants

| Participant | Runs | Sees | Earns |
|---|---|---|---|
| **Client daemon** | Embedding, first dense layers, last layer, LM head, MTP or draft model, prefix cache, route construction, wallet, OpenAI-compatible server on localhost | Everything about its own user, nothing about anyone else | Nothing; spends |
| **Gateway** (new) | The client daemon's job for users who will not install it; a hosted OpenAI endpoint, reachable directly or as an onion service | Plaintext of every request it serves, and the client's IP unless the client uses Tor | A fee; operated by the project inside an enclave, or by a third party the user trusts |
| **Open stage node** | A contiguous layer range on consumer hardware, never the whole model | Hidden states for its layers, a one-time client key, the onion addresses of its two neighbours | Base rate per token served |
| **Attested stage node** (optional add-on) | The same, inside a hardware enclave on server-class hardware, with remote attestation | The same, except that its operator sees none of it | Premium rate |
| **Pod** | A set of mutually vouched nodes on a fast network presenting one or more stages | What its members see | As its members |
| **Relay** (optional) | Forwarding between client and first or last stage, for the relay return mode | An opaque session token, one stage's onion address, and model-state frames until end-to-end encryption across it exists | Per byte |
| **Hub** | Signed node directory; peering with other hubs | Node announcements; directory fetches, which arrive over Tor under a temporary key | Nothing directly |
| **Mint** | Credit issuance, redemption, XMR deposits and withdrawals, bonds | Deposits (an XMR transaction, or a KYC'd USD purchase), blind redemptions, cash-outs | The network fee |
| **Seed** | Content-addressed weight shards | Which shards are fetched, over Tor | Per byte |

Hub and mint are logically separate so that a directory can be mirrored by anyone while money stays with a party that holds reserves.

---

## 4. Node classes and where enclaves exist

The network is built for the open and light classes: consumer hardware that people already own. The attested classes are server hardware, in practice rented in a datacenter, and exist only for the optional attested tier.

| Class | Hardware | Enclave available | Role |
|---|---|---|---|
| **Attested anchor** | Rented or colocated servers: AMD EPYC (SEV-SNP) or Intel Xeon (TDX) with 256 GB or more; optionally NVIDIA H100, H200, B200 or RTX PRO 6000 Blackwell Server Edition in confidential mode | Yes | Attested stages; the premium tier |
| **Attested standard** | The same CPUs with 64 to 128 GB and a confidential-capable GPU or none | Yes | Attested stages with experts in encrypted RAM |
| **Open anchor** | Mac Studio 256 or 512 GB, multi-GPU rigs, Strix Halo boxes | No | Fat open stages, few hops |
| **Open standard** | 64 to 128 GB desktop, 8 GB or more of GPU, 100 Mbps upload | No | Open stages, experts in RAM, attention on GPU |
| **Light** | Anything with a network connection | Not needed | Relay, seed, directory mirror, draft-model host |
| **Client** | 16 GB for the GLM-5.3 head and tail, 32 GB with the MTP layer | Not needed | The user's own machine |

What has an enclave, as of November 2026: AMD EPYC from Milan onward (SEV-SNP), Intel Xeon from Sapphire Rapids onward (TDX), NVIDIA Hopper and Blackwell datacenter GPUs and the RTX PRO 6000 Blackwell Server Edition (confidential computing mode). What does not: consumer Ryzen and Core, every GeForce and Radeon card, Apple Silicon for third-party workloads. A software container or virtual machine protects nothing against the machine's owner, who can read any process's memory.

---

## 5. Trust tiers for routes

The user picks a tier by model name. The daemon refuses to silently downgrade: if the requested tier has no capacity, the request fails with a message, and `/status` always shows which tier and which nodes served the last request.

| Tier | Model name | Stages | What the user gets | What remains |
|---|---|---|---|---|
| **Open** (default) | `glm-5.3` or `glm-5.3:open` | Any admitted node, at least two independent ones per route | Anonymity. Tampering detected by redundant execution (section 7). Cheapest, often free | An open node can, with effort, reconstruct what it was sent |
| **Pod** | `glm-5.3:pod` | Only nodes in pods the user has vouched for | Content protected by trust in people | A friend's compromised machine |
| **Attested** (optional add-on) | `glm-5.3:attested` | Every stage attested; attestation verified by the client before any activation is sent | Operators cannot read content or alter the code. Integrity by construction | Physical attacks on self-hosted enclaves; vendor root of trust; side channels; traffic volume and timing |
| **Ends attested** (optional add-on) | `glm-5.3:ends` | First and last swarm stages attested, middle stages open | Inversion is easiest at the ends, where activations are nearest the embedding and the logits; those ends are closed | Middle-layer inversion still recovers most tokens with effort; steering by a middle node is caught statistically |

Defaults: `glm-5.3` with no suffix means open, because open is the tier that must work on its own. A request with `tools` present is checked on every token rather than a sample (section 7), because a steered output becomes an executed command; the daemon says so in `/status`. The two attested tiers come last in PLAN.md and nothing else may depend on them: a request for a tier that is not available fails with a message.

Pricing is per tier (section 10). Where attested nodes exist, their premium pays for server hardware, and the price difference is the honest statement of what the open tier lacks.

---

## 6. Confidentiality: what nodes can learn

### 6.1 What the research says

A node that holds layers of an open model and receives hidden states can reconstruct the input text. ActInv recovers tokens at 99.8 percent precision when the client holds 2 of 28 blocks of a small model and 77.7 percent when it holds 7 of 28; across model sizes to 30B the figure stays near 100 percent at 5 client blocks. Deeper cuts leak less but still leak under optimisation-based attacks. Gaussian noise only helps at levels that destroy the output. Permutation schemes are broken. Rotation schemes require retraining. Homomorphic encryption and multi-party computation are thousands of times too slow at this scale. Sources in section 18.

**So, on an open node, content confidentiality cannot be guaranteed.** Splitting the model by layer does not change this: every stage receives a full hidden state for every token, and that state encodes the whole prompt. Splitting is not secret sharing. The open tier does everything that is available and says so to the user. What protects an open-tier user is that the node that could reconstruct a prompt cannot learn whose it is, which fails if the prompt itself names its author; the disclosure says that too. The pod tier and the optional attested tier are the answers for anyone who needs more.

### 6.2 What every tier does

| Mechanism | Buys | Does not buy |
|---|---|---|
| Client holds embedding, first dense layers, final layer and LM head | No node sees token IDs, embeddings or logits; output tokens are never on the wire; an attacker must invert rather than look up | Does not stop inversion, raises its cost from trivial to minutes of GPU time per prompt |
| Chain forwarding over Tor | No node knows the client's address, and no node sees the whole route | Does not hide content |
| A fresh key per request | A node that inverts a prompt does not learn whose it is and cannot join it to other requests | Does not hide content; a conversation that opted into a session is joinable for as long as it lasts |
| Chaff sessions (optional) | A node cannot tell which of several sessions is real | Costs credits proportional to the chaff |
| Privacy depth (optional) | The client holds extra MoE layers at 4.8 GB each for GLM-5.3 | Helps at the margin, not in kind |
| Attestation (optional attested tiers) | The operator cannot read memory or change the code | Physical attacks; vendor trust; traffic analysis |

### 6.3 The attested stage (optional add-on)

This is the only known way for a node to prove it cannot read what it serves, and it needs server hardware (section 4). It is built last, and nothing else in this spec may depend on it.

- **Image.** A confidential virtual machine (SEV-SNP or TDX) that boots a minimal, reproducibly built image: firmware, kernel, initramfs, the stage binary, and nothing else. Weights are fetched by content hash into encrypted memory. The launch measurement of this image is published with each release, and anyone can rebuild the image and reproduce the hash.
- **Attestation.** At boot the enclave generates its transport key and asks the platform for a signed report that binds the image measurement to that key. On connection the stage hands the report to the client (or to the previous stage, during route setup). The verifier checks the vendor signature chain, that the platform's firmware versions are not revoked, that the measurement is in the set the client accepts, and that the key in the report is the key on the connection. Only then does any activation flow. GPU attestation is checked the same way when the GPU is inside the enclave boundary.
- **Operator view.** The operator can start and stop the node, see how much traffic it carries and when, and nothing else. The operator's own long-lived key signs the enclave's key once, so that reputation accrues to the operator while content protection comes from the hardware.
- **Cost.** Memory encryption costs a few percent on a memory-bound decode. GPU confidential mode costs more on host-to-device transfers, which mostly affects prefill. Server rental is the real cost, and the premium rate exists to cover it.
- **Physical attacks.** Late 2025 work broke SGX and SEV-SNP on DDR4 with a memory-bus interposer costing tens to hundreds of dollars, and a follow-up reportedly reaches DDR5 and TDX; Intel and AMD state that attacks with physical access are outside their protection boundary. For an enclave rented in a datacenter the operator never touches the hardware and this does not apply. For an enclave in an operator's own rack it does. Where a platform's attestation carries cloud-provider evidence, the directory records it and the client may prefer it; otherwise the residual is disclosed and the usual redundancy sampling continues at a low rate.

### 6.4 The client head and tail

Unchanged from v0.1 section 9.3: for GLM-5.3 the client holds about 8 GB without the MTP layer and 13 GB with it, and runs its share in about 30 ms per token on a CPU-only laptop or a few milliseconds on a GPU. Kimi K3's last layer is a 15 GB MoE layer; clients that cannot hold it must use a pod or attested last stage, and the daemon says so.

### 6.5 Pods and vouching

Unchanged from v0.1 section 9.4. A pod is grown by invitation and mutual signed vouches; the client holds a list of trusted pod identities with a trust level per pod.

---

## 7. Integrity: what a node can do to a request

A node returns numbers. It cannot execute anything on the client; the client treats every byte from the network as hostile, bounds every frame, and its decoders are fuzzed. What a node can do is return wrong numbers: garbage, which the next stage catches on norm and finiteness checks, or subtly steered activations, which change what the model says. For a user who reads the answer this is an annoyance; for an agent that executes the answer it is a real attack.

### 7.1 Compartmentalisation: no node holds a whole model

The client always keeps the embedding, the first layers, the last layer and the output head. The remaining layers of a route are served by at least two nodes with different long-lived keys, and no single key serves more than a capped share of them. The cap is a client setting with a network default, enforced by the client when it builds a route and reported in `/status`; a hub refuses an announcement for a layer range above the cap. Development mode (`--network direct`) may ignore the cap.

What this buys: no node can swap the model for another, see token ids or logits, or run the network's models on its own. What it does not buy, and must never be claimed: a node holding a slice can still add a steering vector to the hidden state it passes on, and every later layer faithfully computes on it. A slice of the layers is enough to bias behaviour. Compartmentalisation limits what one operator controls; redundant execution (below) is what detects tampering. It is also only as strong as the evidence that the keys on a route belong to different operators (section 17).

### 7.2 Detecting tampering on open nodes

On open nodes integrity is statistical unless every token is checked, and the mechanisms are v0.1 section 12 with these additions:

- **Shadow admission.** A new node serves real traffic in parallel with a trusted node holding the same layers; its outputs are compared and discarded until it has produced a configured number of verified tokens.
- **Redundant execution.** The client forks a stage's input to a second, independent node holding the same layers and compares outputs, with a cosine threshold that separates kernel nondeterminism from tampering; a third execution attributes disagreements. Ordinary requests are sampled at 1 to 2 percent of decode tokens, rising for low-reputation nodes. Requests with `tools` present are checked on every token, at roughly double the cost for the checked stages. The LAN measurements found that GLM-4.7-Flash produced identical text at temperature zero across Vulkan, Metal and CPU backends, which is encouraging for cross-hardware comparison, but the threshold still needs data at frontier scale.
- **Signed output logs.** Each stage signs a running hash of its outputs per session, so a dispute can be audited by replaying the inputs on another node.
- **Bonds.** A node posts a bond in credits before it leaves shadow mode. Proven tampering forfeits it. The bond is sized so that the expected gain from tampering at the sampling rate is negative.
- **Attested exemption.** Where attested nodes exist, they skip shadow mode and run at the minimum sampling rate, because their code is proven.

Reputation lives on a node's long-lived key and is public; its credit balance is not.

---

## 8. Anonymity: who is asking and who is serving

### 8.1 Identities

Every client and node is an Ed25519 key. Nodes keep a long-lived key for reputation. There are no accounts, email addresses or usernames anywhere in the system.

**A fresh client identity per request is the default.** Each request uses a new key, a new route, new Tor circuits and new session and route ids; directory lookups use separate temporary keys; nothing is cached on the nodes afterwards. Two requests from one user are as unlinkable as two requests from different users, in the way a Monero stealth address gives every payment a one-time key. The cost is that every turn of a conversation prefills the whole conversation again.

**A conversation may opt into a session.** The user asks for it explicitly, per conversation, through the local API. A session keeps one client key and one route for that conversation, so the nodes keep its prefix cache and its replay log (failover, v0.1 section 7). The stages on the route can then tell that those requests belong together, though still not whose they are. The daemon states this when a session starts and reports it in `/status`; a session ends on request, on idle timeout, or when the daemon stops, and its state on the nodes is then discarded. Sessions are never the default and never started silently.

### 8.2 The chain

The client builds the route from the directory. Each hop's instructions are sealed to that node's key and every transport connection is authenticated, so a hop learns only its two neighbours. A stage sees its route instructions, model states, and traffic metadata. Activations are encrypted on the network but visible inside the stages that compute on them. Returns can follow the chain, an existing client connection, or an optional relay; activations crossing an application relay must be encrypted end to end between the client and the stage, so that the relay carries ciphertext. Content confidentiality against a stage is a trust-tier property (section 5), not a consequence of route sealing.

Per-token cost of the chain: one link crossing per stage plus the client's two legs, instead of a client round trip per stage. A route of eight US datacenter stages spends roughly 150 to 250 ms per token on the network before speculative decoding, which is what the physics budget assumed.

### 8.3 Reaching the swarm

| Mode | Path | Who sees the client's address | Added per token |
|---|---|---|---|
| **Tor** (the network) | Client and stages connect through onion services; no dfin relay required | No dfin peer receives the client's IP; Tor and local-network observers have their own visibility | To be measured over the public Tor network |
| **Tor with a dfin relay** (optional) | Client over Tor to an application relay, then onion stages | No dfin peer receives the client's IP | Additional connections and relay processing |
| **Gateway** | Client to a service running the head and tail, reachable as an onion service | Gateway sees plaintext content, and the client IP unless the client uses Tor | To be measured |
| **Direct** (development only) | QUIC to stages or a dfin relay on a LAN | The directly contacted peers | LAN measurements exist. No anonymity claim; not a product mode |

Anonymity takes priority over latency: failures never select direct networking, and a Tor and a direct endpoint cannot talk to each other. The implementation uses a separately configured trusted Tor daemon and the Tor project's maintained SOCKS protocol library; embedding Tor in the binary (arti) is wanted so that a user installs one thing. Speculative decoding (section 13) is the way to reduce network round trips.

### 8.4 Every node is an onion service

A stage, hub, relay, seed or mint publishes a v3 onion address and listens only on loopback. Its operator's network address appears nowhere: not in the directory, not to its neighbours, not to clients. There is no separate "cloaked" class of node and no relay-allocated public port. Clients and nodes pin a peer's dfin node key inside the Tor connection; reaching the right onion address is not enough.

### 8.5 Money without identity

Credits are blind-signed. The mint sees that an XMR deposit arrived and issues tokens it cannot later recognise; nodes redeem tokens the mint cannot tie to a deposit. A USD purchase is KYC'd by the processor, and that identity still does not reach any usage record. Section 10.

### 8.6 Directory, logs, residuals

Directory fetches use Tor and separate temporary identities. A hub can still personalize the node set it returns, and colluding peers can correlate route choice and timing; clients therefore compare independent directory views (section 12). The directory must carry evidence that identities belong to different operators, because compartmentalisation depends on it, and what that evidence can be is an open question (section 17). Session state is kept in memory, never on disk; diagnostic logs never contain content, keys of other parties or route details. The protocol cannot prevent an operator from recording received states, and the disclosure says so. Tor does not guarantee resistance to broad traffic correlation.

---

## 9. Location

There is no location feature. Users and operators may be anywhere; the network does not verify, require, record or let anyone choose a country or region. An onion service cannot be timed or placed, and that is the point of it.

Region fencing by timing probes was built once and removed on 9 November 2026. Its purpose was to keep a large actor from serving unsafe inference. That purpose is now carried by compartmentalisation (section 7.1) and redundant execution (section 7.2): no node holds a whole model, and what a node does to its slice is checked against an independent node.

The one fact about location that needs no verification: gateways, the first hub and the mint are wherever the project puts its own servers.

---

## 10. Money

### 10.1 Unit

One credit is the work of serving one token through one reference stage, one eighth of GLM-5.3 at IQ4_XS, on the open tier. Consuming one GLM-5.3 token on an open route costs 8 credits at any route length. Attested stages earn and cost a multiplier set by the directory and shown to users before they choose a tier. Prefill tokens cost the same as decode tokens. Reliability and coverage multipliers are as in v0.1 section 11.1.

### 10.2 Two ways in, one way out

- **Serve.** A node earns credits for every token it serves. Spending them on its owner's own use is the free tier; nothing about it touches money.
- **Buy.** A user deposits XMR with the mint and receives credits at the published rate. USD is accepted while a processor will take it; when none will, the network is XMR-only and says so. Credits bought and credits earned are indistinguishable.
- **Cash out.** A node redeems credits for XMR at the published rate minus the network fee, paid out per epoch (daily) to an address it chooses. This is the "percentage of funds proportional to compute served": each node's payout is its share of all tokens served, weighted by tier and multipliers, less the fee that funds the project.

### 10.3 Full reserve, provably

Credits are created only against deposits, plus a bootstrap subsidy with a published cap. The mint therefore always holds enough XMR to redeem every credit in circulation at the current rate. It proves this with Monero reserve proofs, which show that a wallet controls at least a stated amount without exposing its transactions, published alongside the number of credits outstanding. A mint whose reserves fall short is visibly insolvent before anyone loses money.

### 10.4 Paying per token

Credits are Chaumian e-cash in the Cashu style, built on the Cashu Development Kit with an XMR backend in place of Lightning. A client attaches tokens for the request's maximum length to each stage on the route; a stage redeems what it served and returns the rest as fresh blinded change; a stage is paid for a token when the next stage acknowledges plausible output, and the client pays for a token when it receives the final hidden state. Disputes resolve against the party with the worse redundancy record. Denominations are powers of two so that tokens, change and redemptions are small.

### 10.5 What it looks like to an operator

Like a mining pool payout: run the binary, hold a layer range, get paid per token served, in XMR, daily, in proportion to work verified. The same facts that govern mining apply. Electricity is the operator's cost. Batching several sessions in one decode step is the lever: a node serving eight sessions reads its expert weights once per step and earns close to eight times as much. Attested operators carry the cost of server hardware and earn the premium. The network publishes its aggregate earn-to-spend ratio and the daily rate per tier so operators can decide whether to join.

### 10.6 Fraud

Shadow admission before any node earns live-traffic credits. Bonds forfeited on proven tampering. Payment only on delivery. Sampled redundancy. A balance cap per pseudonym, high enough not to matter to honest users, to limit sybil hoarding of the bootstrap subsidy.

### 10.7 Federation

Several mints may exist. Mints peer by trust lines and settle in XMR. A node accepts tokens from any mint its hub trusts; a wallet holds tokens from any mint it has earned at. If a mint disappears its tokens die, which is the incentive to spread balances and for mints to publish reserves.

### 10.8 Law

A mint that takes money and pays operators is a regulated activity in many places. Before the mint opens: counsel on money transmission and tax reporting for payouts, sanctions screening on the USD rail, and the two model licences on redistribution of shards. The XMR rail does not remove these questions; it changes which of them can be enforced.

---

## 11. What is taken from Monero

| Monero component | What it does there | What this design takes |
|---|---|---|
| Tor and I2P integration (`--tx-proxy`, anonymous inbound, onion-reachable remote nodes) | Hides who broadcast a transaction and who runs a node | The whole transport-anonymity layer: Tor as the only network, with no fallback; every node and gateway an onion service; a trusted external Tor daemon today, arti embedded in the binary later |
| Keys, not accounts; stealth addresses | Every payment goes to a one-time key | Per-session keys derived from a master key; no accounts anywhere |
| XMR itself | Private, censorship-resistant money | The payment rail: deposits in, payouts out, reserves held |
| Reserve proofs | Prove a wallet holds funds without revealing spends | The mint's solvency proof |
| Remote-node etiquette | Wallets are told that a remote node sees their IP and queries | Plain disclosures of what each party can see: what an open node could reconstruct, what a session links, what a relay or gateway carries |
| Seed nodes, peer lists, ban lists | Bootstrap and self-defence of the peer network | Directory bootstrap and node admission |
| Reproducible builds, signed releases, no company, research group, community funding | Why it is trusted | Release practice from day one; a published threat model; a standing invitation to audit |
| Ring signatures, confidential amounts, the chain itself | Ledger privacy | Nothing. There is no ledger; the mint's spent list is the only record and it holds unlinkable tokens |
| Dandelion++ | Hides a transaction's origin on a gossip network | Nothing directly; announcements go over Tor instead |

Monero's peer links are not encrypted, because transactions are public. Ours are, because activations are not. Copying Monero's wire protocol would be a mistake; copying its practices and its money is the point.

---

## 12. Coordination and the centre

Three things are centralised at launch and all three are the project's: the directory (hub), the mint, and release signing. They are the deplatformable centre, and the design makes each replaceable rather than pretending it is absent.

- **Directory.** Signed by the hub, mirrored by any light node, fetchable by onion address and from seed nodes compiled into the binary, so no domain is required. Clients cache it and keep routing from the cache if every hub is unreachable. A second hub can be stood up by anyone with the software; hubs peer by exchanging keys.
- **Mint.** Full reserve with public proofs (section 10.3). Federation by trust lines. If the project's mint is shut down, outstanding tokens at that mint die, nodes keep serving against tokens from other mints, and the project's loss is bounded by its reserves.
- **Releases.** Reproducible builds, signatures from several independent keys, mirrors on more than one host, the image measurement for attested nodes published with each release.

What the project's own disappearance would do: existing routes keep working from cached directories; credits at the project's mint die; new nodes find other hubs or none until someone runs one. That is the floor, and it is the same floor Tor and Monero have.

---

## 13. Execution, protocol and failure handling

Pipeline parallelism across the internet, pipeline or expert parallelism inside a pod, routed experts in system RAM with attention and KV on the GPU, FP8 or f16 activations between stages, chunked prefill, micro-batching across sessions, prefix caching and session affinity, three layers of speculative decoding: all as v0.1 section 6. Replay logs and failover as v0.1 section 7, for conversations that opted into a session (section 8.1); client-side routing over a signed directory as v0.1 section 8, with the trust tier and the compartmentalisation cap (section 7.1) added as constraints. Per-layer content-addressed shards as v0.1 section 13. Prefix caching and session affinity apply only inside an opted-in session.

**What has been measured**, on a 30B-A3B model (GLM-4.7-Flash, 18.5 GB at Q4_K_M) on a LAN, log in `archive/phase0-findings.md`:

| Measurement | Result |
|---|---|
| Split exactness | Pipelined output identical to the unsplit model on tiny `llama`, `glm-dsa` and `kimi-k3` models and on the real model, at f16 and f32 activations |
| One machine, four pieces in one process | 20.6 tok/s against 19.8 unsplit |
| Two machines over home WiFi (PC head and tail, MacBook Air layers 1 to 20, PC layers 21 to 45) | 16.2 to 16.8 tok/s, identical text, prefix cache hits across machines |
| Protocol overhead | 0.4 ms per hop on loopback |
| WiFi cost | 15 to 25 ms per token on a link with a 3 ms ping floor |
| Cold start | First request on a freshly started stage took 11 to 27 s while the OS faulted in expert weights; locking weights at load (`--mlock`) brought the first prefill to 1.7 s at the cost of a longer start. Reading weights into private memory made decode twenty to forty times slower on a 32 GB machine through paging |
| Threads | Fewer threads than cores cost 30 to 60 percent on a stage running alone and helped when four pieces shared one CPU |

**Requirements on the protocol that follow from those measurements:**

- **Chain forwarding**, not a client-driven loop (section 8.2). The stage protocol has route setup, next-hop forwarding and a trailer in which each stage reports its compute time so the client still sees per-stage cost.
- **Stage state is as wide as the architecture needs.** GLM-5.3-Flash (`glm5-next`) carries four residual streams between layers; Kimi K3 carries an attention-residual bank that grows with depth. The activation frame's row is `k` times `n_embd` wide with `k` known from the model and the layer index (protocol section 11).
- **Boundary rules per architecture.** `glm-dsa` (GLM-5.3) stages must begin at layers that run the full indexer. Models with KDA layers (Kimi K3) cannot roll a session back, so prefix-cache rewind on those stages means re-prefill from the last snapshot; stages holding KDA layers keep periodic state snapshots.
- **Weights are fetched by layer range.** A node downloads only the shards for its range (v0.1 section 13); the loader must open a sharded file set with its out-of-range shards absent.
- **Locked weights by default.** Stages lock their mapped weights at load and refuse to serve before the lock completes, so the first request is as fast as the hundredth. Thread count defaults to the cores available to that process.
- **Hardening.** Frame size limits per message type, read and idle timeouts, session caps per connection, fuzzed decoders.

---

## 14. API and client daemon

As v0.1 section 14, with: tier suffixes on model names (section 5); the opt-in for a conversation session (section 8.1); `/status` reports the tier, whether a session is open and what it links, the share of layers each node on the last route held, how many tokens were checked by redundant execution, and for each node whether it was open, pod or attested; a wallet balance; and a one-time disclosure shown when an open route is first used. There is no switch that turns Tor off outside development mode. The API listens on numeric loopback only and rejects foreign browser origins. The gateway exposes the same API and the same `/status`, minus the local head and tail.

Tool calls never leave the user's machine as anything but text in the conversation. Execution is the agent framework's job, which is exactly why a request with tools is checked on every token on an open route.

---

## 15. Implementation foundations

What each component is built on. Whether it exists is in [STATUS.md](../STATUS.md), not here.

| Component | Build on |
|---|---|
| Stage runtime | Mainline llama.cpp at a pinned commit plus the layer-range patch (`vendor/README.md`). Whether to keep this or adopt an existing split-inference project is an open evaluation in PLAN.md; the requirement is only that any llama.cpp architecture works as a stage and split output equals unsplit output |
| Transport | Tor onion services reached through SOCKS; inside the Tor stream, TLS 1.3 with certificates bound to node keys, and yamux for multiplexing. QUIC (quinn) with the same key binding for development mode only |
| Chain and relays | Own protocol over the transport (`protocol.md`): sealed per-hop route records, three return modes |
| Tor | A trusted external Tor daemon and `tor-socksproto` today; arti embedded in the binary later |
| Credits | Cashu Development Kit with an XMR backend |
| Monero | A Rust Monero wallet library (monero-serai) or monero-wallet-rpc for deposits, payouts and reserve proofs |
| Directory | Signed records, mirrored by several hubs; no libp2p unless something proves it necessary |
| Weight distribution | Per-layer shards with hashed, signed manifests, served between onion peers over the same transport |
| Client daemon and OpenAI API | Rust, axum: streaming, sampler, chat templates, tool-call parsing |
| Attestation (optional add-on) | AMD SEV-SNP and Intel TDX confidential VMs with measured boot; existing Rust verification libraries for SEV and TDX reports; NVIDIA's attestation service or local verifier for GPUs; dstack and Constellation as references for reproducible confidential images |

---

## 16. Acceptance: how we know dfin does what this says

The order of work and the tasks are in [PLAN.md](PLAN.md). These are the tests the finished system passes; each is repeated on the milestone that delivers it.

1. **Exactness.** Split output equals unsplit output on every supported architecture and on a real model. (Met; see STATUS.md.)
2. **Anonymity.** On a route over the public Tor network, no node learns the client's address or the full route, and two requests from one client cannot be joined by any node unless the user opted into a session.
3. **Compartmentalisation.** The client refuses any route on which one key would hold more than the cap, and `/status` shows each node's share.
4. **Tamper detection.** A deliberately malicious open node is caught and delisted within 1,000 tokens; with `tools` present, a steered token never reaches the caller.
5. **Churn.** Kill a stage mid-generation in an opted-in session and the user sees a pause under 10 s, not an error.
6. **Money.** Strangers join and are paid in XMR; the mint's reserve proof matches credits outstanding; no party can join a purchase to a usage record.
7. **No centre.** Two independent hubs and two mints peer; the network keeps serving with the project's own hub and mint switched off; a third party reproduces the release build byte for byte.
8. **Scale.** GLM-5.3 at 3 tokens per second or better across 4 to 8 machines before speculation, 5 after; cold 20k prefill under 30 s with locked weights; an agent (OpenCode) completes a real coding task.
9. **Attested tier (optional add-on).** An attested route is verified end to end by someone outside the project.

Test 8 is the go or no-go measurement for frontier-size models inherited from the first design. It needs rented hardware (about 400 GB of RAM across machines) and has not been run.

---

## 17. Risks and open questions

**Open-node confidentiality.** Section 6. The product must never claim it and must show the disclosure. The risk it carries is that users will not read it.

**Operator diversity.** Compartmentalisation and redundant execution both assume that two keys mean two operators. One operator can register many keys. Bonds, shadow admission and reputation that takes time to earn raise the cost; none of them proves independence. Until something does, the claim to users is "no single key holds the model", not "no single operator".

**Enclave trust** (optional attested tier). Attestation rests on Intel, AMD and NVIDIA keys and on the absence of a successful physical or side-channel attack on that platform. Both have failed before. The design's response is disclosure, preference for cloud-attested platforms where distinguishable, and continued low-rate redundancy sampling on attested nodes.

**Latency.** The desktop open tier is 2 to 6 tokens per second for agents even with speculation, and Tor adds half a second to two seconds per loop. A request without a session prefills its whole conversation every turn. None of this has been measured over the public Tor network. The honest pitch remains "slow but yours".

**Money.** A mint is a custodian. The reserve proof makes insolvency visible but does not prevent theft or seizure. The USD rail will probably be refused; the plan already tolerates XMR-only. Legal review precedes the mint, not follows it.

**Operator exposure.** Operators on the open tier can, with effort, see content; attested operators cannot. Either way an operator runs a slice of a model that produces whatever users ask. Counsel on relay-style liability, and a plain statement to operators of what they can and cannot see, before strangers are paid to run nodes.

**Licences.** Both models have custom licences; redistributing shards is distribution. Review before any weights leave the core team.

**Centralisation pressure.** Hubs and mints are easiest to run one of. Federation is part of the required system, not an extra.

**Open questions**

1. The cosine threshold for redundancy checks on frontier-scale models across hardware mixes; the identical-text result on a 30B model is encouraging but not sufficient.
2. The compartmentalisation cap: what share of the swarm's layers one key may hold, and how it trades against hop count and latency.
3. What evidence of operator independence a directory can carry without identifying operators.
4. The MTP acceptance rate for GLM-5.3 and Kimi K3 with the shipped heads, which decides whether the network delivers 1 or 2 tokens per second over Tor.
5. Whether arti's onion-service support is good enough to embed, or the external daemon stays.
6. A workable Kimi K3 tail for 16 GB clients.
7. Whether USD purchases should exist at all, given that they are the only KYC'd path and the one most likely to be refused.
8. For the optional attested tier: the premium that clears the market, and whether cloud-provider evidence in attestation reports is distinguishable enough to rank platforms.

---

## 18. Sources

**Measurements (this repository)**
- [archive/phase0-findings.md](archive/phase0-findings.md): stage-boundary state per architecture, exactness, one-machine and two-machine results, cold start.

**Privacy and inversion**
- ActInv, "What Does the Server See?": https://arxiv.org/abs/2605.23158
- Dual-sided leakage of prompts and responses in split LLMs: https://arxiv.org/abs/2606.14210
- "Depth Gives a False Sense of Privacy": https://arxiv.org/abs/2507.16372
- Model inversion in split learning for LLMs: https://arxiv.org/abs/2501.05965
- Attack on permutation-based schemes: https://arxiv.org/abs/2505.18332

**Enclaves**
- Battering RAM and WireTap, physical attacks on SGX and SEV-SNP: https://www.techspot.com/news/109712-trusted-enclaves-intel-amd-shown-vulnerable-physical-attacks.html and https://hacken.io/insights/wiretap-and-battering-ram-risks/
- WireTap: https://wiretap.fail
- GPU and CPU enclave requirements, including RTX PRO 6000 confidential mode: https://superprotocol.com/resources/gpu-cpu-tee-requirements
- SpliTEE, enclaves plus differential privacy for LLM inference: https://deeplearn.org/arxiv/840762/splitee:-fast-and-private-llm-inference-by-coupling-gpu-assisted-trusted-execution-environments-with-differential-privacy

**Anonymity and money**
- Tor circuit and onion-service design (the model for chains and onion-service nodes): https://spec.torproject.org/
- arti, Tor in Rust: https://gitlab.torproject.org/tpo/core/arti
- Monero anonymity networks and remote-node guidance: https://docs.getmonero.org/
- Monero reserve proofs: `get_reserve_proof` in the monero-wallet-rpc documentation
- Cashu protocol and Cashu Development Kit: https://docs.cashu.space/protocol and https://github.com/cashubtc/cdk
- Cross-mint theft analysis (January 2026): https://eprint.iacr.org/2026/2174.pdf

**Models, hardware and prior art**: [archive/design-v0.1.md](archive/design-v0.1.md) section 18 and [prior-art-2026-11.md](prior-art-2026-11.md).
