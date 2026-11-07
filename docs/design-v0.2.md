# Decentralized Frontier Inference Network

**Design v0.2, 6 November 2026.** Supersedes v0.1 (5 November 2026) on goals, threat model, participants, trust, anonymity, money and phasing. The model physics (v0.1 sections 2 and 3), execution model (6), failure handling (7), routing (8), weight distribution (13) and API (14) stand as written there except where this document says otherwise; they are summarised in section 13 here rather than repeated. Phase 0 measurements that inform this revision are in [phase0-findings.md](phase0-findings.md).

A network that runs frontier-scale open-weight models, Chinese-lab models first (GLM-5.3, then Kimi K3), on machines owned by many unrelated people, and exposes them through an OpenAI-compatible endpoint. It exists so that these models stay reachable from outside the labs' own endpoints and outside any single cloud, however unpopular hosting them becomes. Users are anonymous to the machines that serve them. Nodes either provably cannot read what they serve (attested tier) or are told to the user as nodes that could (open tier). Operators are paid, in Monero, in proportion to the work that runs through their machines.

---

## 0. What changed since v0.1, and why

| Change | v0.1 | v0.2 | Why |
|---|---|---|---|
| Adversary | A curious node, a steering hub | **Deplatforming**: clouds, payment processors, model hosts and code hosts refusing service, and pressure on identifiable operators | The reason the network should exist is that hosted endpoints for these models may disappear |
| Who runs nodes first | Friends, invite-only pods, strangers later | **Strangers from the start**, running nodes for pay, the way mining works | The operator does not run the compute. The network is other people's machines |
| Money | Credits only, "no token, no money" | **Credits bought with XMR** (USD while a processor allows it), earned by serving, redeemed for XMR. Reciprocal use stays free | Electricity is not free; incentive is what Petals lacked |
| Node classes | Consumer hardware only | **Attested** nodes (hardware enclaves on server-class CPUs and datacenter GPUs) and **open** nodes (consumer hardware) | v0.1 wrote enclaves off because it assumed consumer hardware. Server hardware is allowed, and enclaves are the only known way a node can prove it cannot read what it serves |
| Route trust | Pod-only or public | **Four tiers**: pod, attested, ends-attested, open; chosen by model-name suffix; the client reports which served each request | Users must be able to pay for a guarantee or knowingly accept its absence |
| Jurisdiction | Not addressed | **No network rule.** Region is a per-user preference, enforced by timing probes on direct nodes, impossible for cloaked nodes | Security-conscious users will not disclose location; operators may hide; the front door is in the US by construction |
| Pipeline shape | Client calls each stage in turn (what Phase 0 built) | **Chain forwarding**: stage hands to stage; the client touches only the first and last through a relay | Required for anonymity, and the largest latency win |
| Reference systems | Petals, BloomBee, Tor, Cashu | Add **Monero** as the model for keys-not-accounts, money, Tor integration and release practice. Nothing from its ledger cryptography | Already trusted by the users this is for |
| Stage state | One hidden-state tensor per token | **A row as wide as the architecture needs** per token, a multiple of `n_embd` known from the layer index | GLM-5.3-Flash's hyper-connections and Kimi K3's attention-residual bank carry more than one residual stream between layers (Phase 0 finding) |
| Phases | One pod, friends, open swarm, anchors | Foundation (encrypted chain), **trust and money together**, then scale | Attested operators are paid professionals; they do not appear before money exists |

---

## 1. Goals, non-goals and constraints

**Goals**

1. Serve GLM-5.3 and, later, Kimi K3 through an OpenAI-compatible `/v1/chat/completions` endpoint with streaming and tool calling, from machines that are neither the model labs' own endpoints nor any single cloud.
2. Anyone can run a node and earn. Anyone can use the network either by contributing compute or by paying, in Monero, without an account.
3. No node, relay, hub, mint or payment links a request to a person or an address.
4. A node cannot read a user's content or alter the model's behaviour without that being either impossible (attested tier) or detectable (open tier), and the user always knows which tier served them.
5. No single party whose removal stops the network, including the operator of the first directory and mint.
6. Node churn mid-generation produces a pause, not an error.

**Non-goals for v1**

- Hosted-API latency. Section 13 gives the measured and projected numbers.
- Cryptographic confidentiality on open consumer nodes. Section 6 explains why it is not available and what the open tier does instead.
- Location guarantees for anonymous participants. Section 9.
- Resistance to a global passive adversary. Tor mode is offered for users who need more; it is not the default.
- Vision input, fine-tuning, training.

**Constraints** are those of v0.1 section 1 (consumer latency, upload bandwidth, memory bandwidth, model licences), plus what Phase 0 measured: the protocol itself costs about 0.4 ms per hop; a home WiFi link costs 15 to 25 ms per token; a stage must lock its weights at load or its first request takes ten to thirty seconds.

---

## 2. Threat model

| Adversary | Wants | Can do | What the design does |
|---|---|---|---|
| **Deplatformer**: a cloud, a payment processor, a model host, a code host, a registrar, an app store | The service gone | Refuse hosting and payments, delist weights, take down repositories and domains, pressure identifiable operators | No cloud dependency. Nodes on many small providers and in homes. Weights seeded peer to peer under signed manifests. Software mirrored, reproducibly built, signed by several keys, discoverable without DNS (seed lists in the binary, onion addresses). Directories and mints federated. The XMR rail cannot be refused. Cloaked nodes for operators who want to be unfindable |
| **Open node**, curious or malicious | Prompts, identities, steered outputs, pay without work | Invert the activations it receives (section 6.1), return wrong or steered activations, return garbage, claim work it did not do | Client holds embedding, first layers, last layer and output head, so no node sees text or logits. Relays or Tor keep addresses away. Chaff sessions. Sampled redundancy and shadow admission catch tampering. Payment per token on delivery. Bonds. The user is told in plain words what an open node can do |
| **Attested node operator** with physical access | The same, through the enclave | Memory-bus interposer attacks have broken SGX and SEV-SNP on DDR4 and reportedly reach DDR5 and TDX; vendors call physical attacks out of scope | Attestation still proves the code. Residual physical risk is disclosed. Platforms whose attestation carries cloud-provider evidence rank above self-hosted enclaves. Redundancy sampling continues at a low rate |
| **Relay operator** | Client addresses | See a client IP, a session token, ciphertext, the address of the first or last stage | The client chooses its relay. Tor mode puts Tor between client and relay. Relays hold no content keys and keep no logs |
| **Hub or mint operator** (the project, at first) | Everything a central party could want | Censor nodes from the directory, steal reserves, link a purchase to a buyer | Clients build routes themselves, so the hub cannot steer. Several hubs and mints. Full-reserve mint with Monero reserve proofs. Blind signatures unlink purchase from use. Everything open source and reproducible |
| **Network observer** (ISP, backbone) | Who talks to whom | Timing correlation | Every link encrypted. Tor mode for users who need it. Not defended against globally |
| **Malicious client** | Free service, denial of service, probing nodes | Flood a stage, not pay, send malformed frames | Prepaid per token. Session and frame limits. Rate limits per connection. Parsers fuzzed |
| **Sybil operator** | Many fake nodes to grab routes, harvest activations, farm credits | Register thousands of keys | Shadow admission with verified work. Bonds. Reputation that takes time to earn. Routes prefer long-lived keys. Chaff makes harvested sessions ambiguous |
| **Model licensors** | Their terms honoured | Legal action over redistribution | Counsel reviews both licences before weights are shipped to anyone outside the core team (section 17) |

**What never exists in one place.** No participant holds both a client IP and request content. No participant holds both a route and content. No participant holds both a purchase identity and a usage record. The one deliberate exception is the gateway (section 3), which exists for users who will not run the daemon and is therefore a trusted or attested party by definition.

---

## 3. Participants

| Participant | Runs | Sees | Earns |
|---|---|---|---|
| **Client daemon** | Embedding, first dense layers, last layer, LM head, MTP or draft model, prefix cache, route construction, wallet, OpenAI-compatible server on localhost | Everything about its own user, nothing about anyone else | Nothing; spends |
| **Gateway** (new) | The client daemon's job for users who will not install it; a hosted OpenAI endpoint, reachable directly or as an onion service | Plaintext of every request it serves, and the client's IP unless the client uses Tor | A fee; operated by the project inside an enclave, or by a third party the user trusts |
| **Open stage node** | A contiguous layer range on consumer hardware | Hidden states for its layers, a per-session pseudonym, the addresses of its two neighbours or their relays | Base rate per token served |
| **Attested stage node** | The same, inside a hardware enclave on server-class hardware, with remote attestation | The same, except that its operator sees none of it | Premium rate |
| **Pod** | A set of mutually vouched nodes on a fast network presenting one or more stages | What its members see | As its members |
| **Relay** | Encrypted forwarding between client and first or last stage; introduction points for cloaked nodes | Client IP, an opaque session token, one stage address | Per byte |
| **Hub** | Signed node directory; peering with other hubs | Node announcements; directory fetches by relay IP | Nothing directly |
| **Mint** | Credit issuance, redemption, XMR deposits and withdrawals, bonds | Deposits (an XMR transaction, or a KYC'd USD purchase), blind redemptions, cash-outs | The network fee |
| **Seed** | Content-addressed weight shards | Which shards are fetched, by relay IP | Per byte |

Hub and mint are logically separate so that a directory can be mirrored by anyone while money stays with a party that holds reserves.

---

## 4. Node classes and where enclaves exist

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
| **Pod** | `glm-5.3:pod` | Only nodes in pods the user has vouched for | Content protected by trust in people | A friend's compromised machine |
| **Attested** | `glm-5.3:attested` | Every stage attested; attestation verified by the client before any activation is sent | Operators cannot read content or alter the code. Integrity by construction | Physical attacks on self-hosted enclaves; vendor root of trust; side channels; traffic volume and timing |
| **Ends attested** | `glm-5.3:ends` | First and last swarm stages attested, middle stages open | Inversion is easiest at the ends, where activations are nearest the embedding and the logits; those ends are closed | Middle-layer inversion still recovers most tokens with effort; steering by a middle node is caught statistically |
| **Open** | `glm-5.3:open` | Any node | Cheapest, often free | An open node can, with effort, reconstruct what it was sent, and could steer output until caught |

Defaults: `glm-5.3` with no suffix means attested. Agent frameworks that execute tool calls should use attested or pod; the daemon warns once when an open route is used with `tools` present, because a steered output becomes an executed command.

Pricing is per tier (section 10). The premium is what pays for server hardware, and the price difference is the honest statement of what the open tier lacks.

---

## 6. Confidentiality: what nodes can learn

### 6.1 What the research says

A node that holds layers of an open model and receives hidden states can reconstruct the input text. ActInv recovers tokens at 99.8 percent precision when the client holds 2 of 28 blocks of a small model and 77.7 percent when it holds 7 of 28; across model sizes to 30B the figure stays near 100 percent at 5 client blocks. Deeper cuts leak less but still leak under optimisation-based attacks. Gaussian noise only helps at levels that destroy the output. Permutation schemes are broken. Rotation schemes require retraining. Homomorphic encryption and multi-party computation are thousands of times too slow at this scale. Sources in section 18.

**So, on an open node, content confidentiality cannot be guaranteed.** The open tier does everything that is available and says so to the user. The attested tier is the answer for anyone who needs more.

### 6.2 What every tier does

| Mechanism | Buys | Does not buy |
|---|---|---|
| Client holds embedding, first dense layers, final layer and LM head | No node sees token IDs, embeddings or logits; output tokens are never on the wire; an attacker must invert rather than look up | Does not stop inversion, raises its cost from trivial to minutes of GPU time per prompt |
| Chain forwarding with relayed ends | No node knows the client's address, and no node sees the whole route | Does not hide content |
| Per-session pseudonyms | A node that inverts a prompt does not learn whose it is and cannot join it to other sessions | Does not hide content |
| Chaff sessions (optional) | A node cannot tell which of several sessions is real | Costs credits proportional to the chaff |
| Privacy depth (optional) | The client holds extra MoE layers at 4.8 GB each for GLM-5.3 | Helps at the margin, not in kind |
| Attestation (attested tiers) | The operator cannot read memory or change the code | Physical attacks; vendor trust; traffic analysis |

### 6.3 The attested stage

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

A node returns numbers. It cannot execute anything on the client; the client treats every byte from the network as hostile, bounds every frame, and its decoders are fuzzed. What a node can do is return wrong numbers: garbage, which the next stage catches on norm and finiteness checks, or subtly steered activations, which change what the model says. For a user who reads the answer this is an annoyance; for an agent that executes the answer it is a real attack, which is why agents default to attested routes.

On open nodes integrity is statistical, never absolute, and the mechanisms are v0.1 section 12 with two additions:

- **Shadow admission.** A new node serves real traffic in parallel with a trusted node holding the same layers; its outputs are compared and discarded until it has produced a configured number of verified tokens.
- **Sampled redundancy.** For 1 to 2 percent of decode tokens the client forks a stage's input to a second node and compares outputs, with a cosine threshold that separates kernel nondeterminism from tampering; a third execution attributes disagreements. Sampling rises for low-reputation nodes. Phase 0 found that GLM-4.7-Flash produced identical text at temperature zero across Vulkan, Metal and CPU backends, which is encouraging for cross-hardware comparison, but the threshold still needs data at frontier scale.
- **Signed output logs.** Each stage signs a running hash of its outputs per session, so a dispute can be audited by replaying the inputs on another node.
- **Bonds** (new). A node posts a bond in credits before it leaves shadow mode. Proven tampering forfeits it. The bond is sized so that the expected gain from tampering at the sampling rate is negative.
- **Attested exemption** (new). Attested nodes skip shadow mode and run at the minimum sampling rate, because their code is proven.

Reputation lives on a node's long-lived key and is public; its credit balance is not.

---

## 8. Anonymity: who is asking and who is serving

### 8.1 Identities

Every client and node is an Ed25519 key. Clients derive a fresh per-session key from a master key, so sessions are unlinkable, in the way a Monero stealth address gives every payment a one-time key. Nodes keep a long-lived key for reputation. There are no accounts, email addresses or usernames anywhere in the system.

### 8.2 The chain

The client builds the route from the directory, then sets it up the way Tor builds a circuit: it negotiates a key with the first stage through the relay, then through that encrypted hop negotiates with the second, and so on. Each stage learns the address and key of its two neighbours and nothing else. The last stage returns output to the client's relay under a session token, never to the client's address. Activations are encrypted per link; they must be in the clear inside each stage, which is why confidentiality is a tier property (section 5), while the route itself is hidden from every stage.

Per-token cost of the chain against Phase 0's measurements: one link crossing per stage plus the two relay legs, instead of a client round trip per stage. A route of eight US datacenter stages spends roughly 150 to 250 ms per token on the network before speculative decoding, which is what the physics budget assumed.

### 8.3 Reaching the swarm

| Mode | Path | Who sees the client's address | Added per token |
|---|---|---|---|
| **Relay** (default) | Client to a relay it chooses, relay to first stage; last stage to the same or another relay, relay to client | The relay only | Tens of milliseconds |
| **Tor** (strict) | Client over Tor to the relay | Nobody; the relay sees a Tor exit or rendezvous point | About half a second with a directly reachable relay; one to two seconds with a relay that is an onion service |
| **Gateway** | Client over Tor or directly to a gateway that runs the head and tail | The gateway, unless Tor is used; the gateway also sees plaintext | Tor's cost applies only to the first token and streaming, not to the per-token loop, because the loop runs at the gateway |

Tor's cost per token is the reason it is the strict mode and not the default. Speculative decoding (MTP plus a client draft model, v0.1 section 6.5) multiplies tokens per loop and is therefore worth more in Tor mode than anywhere else. The Tor client is the Rust implementation, arti, with the C daemon as a fallback where arti's onion-service support is not yet enough.

### 8.4 Cloaked nodes

An operator who does not want neighbours to learn their address keeps persistent outbound connections to two relays and is reachable only through them, like an onion service. Each cloaked link adds one relay hop, which the router accounts for. Cloaked nodes cannot be placed in any region (section 9) and are excluded from region-constrained routes; otherwise they earn the same as direct nodes.

### 8.5 Money without identity

Credits are blind-signed. The mint sees that an XMR deposit arrived and issues tokens it cannot later recognise; nodes redeem tokens the mint cannot tie to a deposit. A USD purchase is KYC'd by the processor, and that identity still does not reach any usage record. Section 10.

### 8.6 Directory, logs, residuals

Directory fetches go through a relay or Tor so the hub does not learn client addresses. Relays and nodes keep no session records. A hub that was also a relay that was also a first stage would see IP, pseudonym and hidden states together; the directory flags operators running several roles and the router never uses one operator in two roles on one route. Timing correlation by a well-placed observer is not defended against outside Tor mode.

---

## 9. Location

Users and operators may be anywhere. The network does not verify, require or record a country for anyone. What it offers:

- **The front door is where the project puts it.** Gateways, the first relays, the first hub and the mint are operated by the project on US infrastructure. That is a fact about the project's own servers, and it needs no verification.
- **Direct nodes can be fenced by timing** (done 6 November 2026, phase 1 protocol section 14). Several probe points measure round-trip time to each direct node. Light in fibre allows roughly 100 km per millisecond of round trip, so a node that answers both an east-coast and a west-coast US probe in under about 50 ms is in North America and nowhere else. The directory records the fenced region as a claim with the probe evidence. This bounds where the node's network interface is, not where the computation runs; a fenced node could forward activations elsewhere. Attestation closes that gap, since the attested image does not forward. Fenced plus attested is therefore the only strong location statement the network can make.
- **Cloaked nodes have no location.** That is the point of cloaking.
- **Region is a user preference**, not a rule: "direct nodes fenced to these regions only", off by default. The daemon reports which nodes on a route were fenced, attested, both or neither.

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
| Tor and I2P integration (`--tx-proxy`, anonymous inbound, onion-reachable remote nodes) | Hides who broadcast a transaction and who runs a node | The whole transport-anonymity layer: relays and cloaked nodes modelled on onion services, Tor as strict mode, nodes and gateways reachable as onion services, implemented with arti |
| Keys, not accounts; stealth addresses | Every payment goes to a one-time key | Per-session keys derived from a master key; no accounts anywhere |
| XMR itself | Private, censorship-resistant money | The payment rail: deposits in, payouts out, reserves held |
| Reserve proofs | Prove a wallet holds funds without revealing spends | The mint's solvency proof |
| Remote-node etiquette | Wallets are told that a remote node sees their IP and queries | The relay model and its disclosure: a relay sees your IP, nothing else |
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

Pipeline parallelism across the internet, pipeline or expert parallelism inside a pod, routed experts in system RAM with attention and KV on the GPU, FP8 or f16 activations between stages, chunked prefill, micro-batching across sessions, prefix caching and session affinity, three layers of speculative decoding: all as v0.1 section 6. Replay logs and failover as v0.1 section 7; client-side routing over a signed directory as v0.1 section 8, with the trust tier and region preference added as constraints. Per-layer content-addressed shards as v0.1 section 13.

**What Phase 0 established**, on a 30B-A3B model (GLM-4.7-Flash, 18.5 GB at Q4_K_M), log in `phase0-findings.md`:

| Measurement | Result |
|---|---|
| Split exactness | Pipelined output identical to the unsplit model on tiny `llama`, `glm-dsa` and `kimi-k3` models and on the real model, at f16 and f32 activations |
| One machine, four pieces in one process | 20.6 tok/s against 19.8 unsplit |
| Two machines over home WiFi (PC head and tail, MacBook Air layers 1 to 20, PC layers 21 to 45) | 16.2 to 16.8 tok/s, identical text, prefix cache hits across machines |
| Protocol overhead | 0.4 ms per hop on loopback |
| WiFi cost | 15 to 25 ms per token on a link with a 3 ms ping floor |
| Cold start | First request on a freshly started stage took 11 to 27 s while the OS faulted in expert weights; locking weights at load (`--mlock`) brought the first prefill to 1.7 s at the cost of a longer start. Reading weights into private memory made decode twenty to forty times slower on a 32 GB machine through paging |
| Threads | Fewer threads than cores cost 30 to 60 percent on a stage running alone and helped when four pieces shared one CPU |

**Changes to the protocol that follow from Phase 0 and from this revision:**

- **Chain forwarding** replaces the client-driven loop (section 8.2). The stage protocol gains route setup, next-hop forwarding and a trailer in which each stage reports its compute time so the client still sees per-stage cost.
- **Stage state is as wide as the architecture needs.** GLM-5.3-Flash (`glm5-next`) carries four residual streams between layers; Kimi K3 carries an attention-residual bank that grows with depth. The activation frame's row is `k` times `n_embd` wide with `k` known from the model and the layer index, and the loader's refusal to split these models is lifted (done 6 November 2026, phase 1 protocol section 13).
- **Boundary rules per architecture.** `glm-dsa` (GLM-5.3) stages must begin at layers that run the full indexer. Models with KDA layers (Kimi K3) cannot roll a session back, so prefix-cache rewind on those stages means re-prefill from the last snapshot; stages holding KDA layers keep periodic state snapshots.
- **Weights are fetched by layer range.** A node downloads only the shards for its range (v0.1 section 13); the loader must open a sharded file set with its out-of-range shards absent.
- **Locked weights by default.** Stages lock their mapped weights at load and refuse to serve before the lock completes, so the first request is as fast as the hundredth. Thread count defaults to the cores available to that process.
- **Hardening.** Frame size limits per message type, read and idle timeouts, session caps per connection, fuzzed decoders.

---

## 14. API and client daemon

As v0.1 section 14, with: tier suffixes on model names (section 5); `/status` reports the tier, and for each node on the last route whether it was attested, fenced, cloaked or open; a Tor toggle; a wallet balance; and a one-time disclosure shown when an open route is first used. The gateway exposes the same API and the same `/status`, minus the local head and tail.

Tool calls never leave the user's machine as anything but text in the conversation. Execution is the agent framework's job, which is exactly why the daemon warns when tools and an open route meet.

---

## 15. Implementation foundations

| Component | Build on | Status |
|---|---|---|
| Stage runtime | Mainline llama.cpp at a pinned commit plus the layer-range patch (`vendor/README.md`) | Built and verified in Phase 0 on Windows and macOS; Linux unverified; multi-tensor stage state and sharded loading not yet done |
| Transport | QUIC (quinn) with TLS 1.3 using certificates bound to node keys, or Noise (snow) over QUIC | Not started; TCP today |
| Chain and relays | Own protocol over the transport, telescoped setup modelled on Tor circuits | Not started |
| Tor | arti (Rust); C Tor daemon as fallback for onion services if needed | Not started |
| Attestation | AMD SEV-SNP and Intel TDX confidential VMs with measured boot; existing Rust verification libraries for SEV and TDX reports; NVIDIA's attestation service or local verifier for GPUs; dstack and Constellation as references for reproducible confidential images | Not started |
| Credits | Cashu Development Kit with an XMR backend | Not started |
| Monero | A Rust Monero wallet library (monero-serai) or monero-wallet-rpc for deposits, payouts and reserve proofs | Not started |
| Directory | Signed records, mirrored; libp2p only if NAT traversal proves necessary | Not started |
| Weight distribution | BitTorrent v2 style over the same transport, per-layer shards, signed manifests | Not started |
| Client daemon and OpenAI API | Rust, axum; present today with streaming, prefix cache, sampler | Built; tool-call parsing and chat-template handling for thinking models missing |

---

## 16. Phases

| Phase | Scope | Exit criteria |
|---|---|---|
| **0. Mechanism** (done, 5 to 6 November 2026) | Layer-range stage runtime, client head and tail, OpenAI API, prefix cache, local pod, two-machine pod | Met: exact split on three architectures and a real 30B model; two machines over WiFi at 85 percent of single-machine speed. **Not met**: the GLM-5.3 scale test, which needs about 400 GB of rented RAM and sharded loading; deferred to phase 3 |
| **1. Encrypted chain** | Transport with node keys; chain forwarding with telescoped routes; relays and cloaked nodes; signed directory; per-layer shards and peer seeding; multi-tensor stage state; replay-log failover; Linux build and container image; region fencing by probes; hardening | Three pods in three cities serve one model over the WAN through relays; kill a stage mid-generation and the user sees a pause under 10 s; no node on a route learns the client's address or the full route; a third party reproduces the build |
| **2. Trust and money** | Attested stage image and client-side verification; trust tiers in the API; shadow admission, sampled redundancy, signed logs, bonds; mint with XMR (and USD if a processor accepts), wallets, per-token payment, daily payouts, reserve proofs; disclosures; legal review before the mint opens | Strangers join and are paid; a deliberately malicious open node is caught and delisted within 1k tokens; an attested route is verified end to end by someone outside the project; the mint's reserve proof matches credits outstanding |
| **3. Scale** | GLM-5.3 IQ4_XS across rented attested servers and across open desktops; MTP and client draft model; OpenCode completes a real coding task; Kimi K3 via anchor pods; hub and mint federation; Tor mode measured | GLM-5.3 at 3 tok/s or better across 4 to 8 machines on an attested route before speculation, 5 after; cold 20k prefill under 30 s with locked weights; two independent hubs and two mints peering |

Phase 3's first exit criterion is the go or no-go gate inherited from v0.1. It moved because the hardware it needs is rented, not owned, and because the network layer is now the critical path.

---

## 17. Risks and open questions

**Open-node confidentiality.** Section 6. The product must never claim it, must price the attested tier above the open tier, and must show the disclosure. The risk it carries is that users will not read it.

**Enclave trust.** Attestation rests on Intel, AMD and NVIDIA keys and on the absence of a successful physical or side-channel attack on that platform. Both have failed before. The design's response is disclosure, preference for cloud-attested platforms where distinguishable, and continued low-rate redundancy sampling on attested nodes.

**Latency.** The desktop open tier is 2 to 6 tokens per second for agents even with speculation; attested datacenter routes should be faster, and Tor mode is slower by half a second to two seconds per loop. The honest pitch for the open tier remains "slow but yours".

**Money.** A mint is a custodian. The reserve proof makes insolvency visible but does not prevent theft or seizure. The USD rail will probably be refused; the plan already tolerates XMR-only. Legal review precedes the mint, not follows it.

**Operator exposure.** Operators on the open tier can, with effort, see content; attested operators cannot. Either way an operator runs a slice of a model that produces whatever users ask. Counsel on relay-style liability, and a plain statement to operators of what they can and cannot see, before phase 2.

**Licences.** Both models have custom licences; redistributing shards is distribution. Review before any weights leave the core team.

**Centralisation pressure.** Hubs and mints are easiest to run one of. Federation is built in phase 2, not deferred.

**Open questions**

1. The cosine threshold for redundancy checks on frontier-scale models across hardware mixes; Phase 0's identical-text result on a 30B model is encouraging but not sufficient.
2. The MTP acceptance rate for GLM-5.3 and Kimi K3 with the shipped heads, which decides whether Tor mode is 1 or 2 tokens per second.
3. Whether arti's onion-service support is enough for cloaked nodes and gateways by phase 1, or the C daemon is needed.
4. The attested-tier premium that clears the market: it must cover server rental and still be cheaper than the labs' own endpoints.
5. Whether cloud-provider evidence in attestation reports is distinguishable enough across providers to rank platforms, or whether the network should simply disclose "self-hosted enclave" as a flag set by the operator.
6. A workable Kimi K3 tail for 16 GB clients.
7. Whether USD purchases should exist at all, given that they are the only KYC'd path and the one most likely to be refused.

---

## 18. Sources

**Phase 0 (this repository)**
- [phase0-findings.md](phase0-findings.md): stage-boundary state per architecture, exactness, one-machine and two-machine results, cold start.

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
- Tor circuit and onion-service design (the model for chains and cloaked nodes): https://spec.torproject.org/
- arti, Tor in Rust: https://gitlab.torproject.org/tpo/core/arti
- Monero anonymity networks and remote-node guidance: https://docs.getmonero.org/
- Monero reserve proofs: `get_reserve_proof` in the monero-wallet-rpc documentation
- Cashu protocol and Cashu Development Kit: https://docs.cashu.space/protocol and https://github.com/cashubtc/cdk
- Cross-mint theft analysis (January 2026): https://eprint.iacr.org/2026/2174.pdf

**Models, hardware and prior art**: as v0.1 section 18.
