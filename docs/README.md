# Documents

Start with `../STATUS.md` for what exists and what does not.

| Document | What it is |
|---|---|
| `anonymity.md` | Strict Tor defaults: how requests work, setup, boundaries, direct-mode migration, verification, and what is reused from Monero/Tor. |
| `anonymity-work-log.md` | Dated implementation decisions, checks, results, and outstanding work for strict anonymity. |
| `design-v0.2.md` | The current design: threat model (deplatforming), participants, node classes and where enclaves exist, trust tiers, confidentiality, integrity, anonymity, location, money, what is taken from Monero, phases 1 to 3 with exit criteria, risks. Read this first. |
| `design-v0.1.md` | The first design. Superseded on trust and money, still the reference for the physics: model sizes, what crosses a stage boundary, bandwidth and memory per session, replay logs and failover (section 7), routing (section 8), shards (section 13). |
| `phase1-protocol.md` | The wire contract of the encrypted chain: identities, transport and limits, messages, route setup, return modes, relay, directory, cloaked nodes, shards, wide stage state, region fencing, fuzzing, then a status and what changed from Phase 0. |
| `phase0-findings.md` | What was learned building the LAN pod: what each architecture needs at a stage boundary, the patch, tiny-model and real-model results, the two-machine run, cold start. Dated entries. |
| `phase1-findings.md` | What was learned building the chain: measurements on a real model, lessons from each piece (directory, cloaked nodes, shards and the f16 finding, wide state, fencing, fuzzing), and what is still open. |
| `prior-art-2026-11.md` | What else exists (Petals, Mesh LLM, prima.cpp, BloomBee, Unsloth's patch) and the state of llama.cpp for the target models, as of November 2026. |
| `mac-node-prompt.md` | A self-contained prompt for setting up a second machine (a MacBook) as a stage, written for another Claude session to carry out. |
| `logs/` | Raw benchmark and verification logs the findings cite, named by what and when. |
