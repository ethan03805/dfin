# Plan

The order of work for dfin. [SPEC.md](SPEC.md) says what dfin must do, and all of it is
required; this page says in what order it gets built and how each piece is known to be
finished. [STATUS.md](../STATUS.md) says what exists today.

## How to use this page

- Take the first unticked task in the lowest-numbered milestone whose **Needs** you
  have. If the owner has asked for something else, do that instead.
- One task, one branch, one pull request. The same pull request ticks the task here and
  updates STATUS.md. A task is not done until its **Done when** line is true and CI is
  green.
- Milestones are an order, not versions. Nothing here is optional except milestone 7,
  which the spec itself marks as an add-on.
- If a task turns out to conflict with SPEC.md, stop and ask the owner. Do not resolve
  it by editing the spec.

## What needs the owner

An agent prepares these and stops; the owner decides.

| Gate | Applies to |
|---|---|
| Replacing the inference core | Task 1.1's recommendation |
| Running commands on the owner's other machines | Milestone 2 |
| Renting hardware | Tasks 6.1 and 6.2, milestone 7 |
| Real XMR, a real mint, anything with funds | Milestone 4 stays on Monero stagenet until legal review is done |
| Shipping model weights or shards to anyone else | Licence review first (SPEC section 17) |
| Any public deployment: a hub, relay, seed or gateway others can reach | Every milestone |

## 0. One repository, one description (done 9 November 2026)

- [x] Tor-by-default work merged to `main`; stale branches deleted.
- [x] One spec (`docs/SPEC.md`), one plan (this page), one status page, one set of rules
      for agents (`AGENTS.md`); earlier design documents and work logs moved to
      `docs/archive/`.
- [x] UDP cloaking, region fencing and `dfin probe` removed from code, tests, fuzz
      targets and the protocol.
- [x] `dfin bench` works again (it sent a model name the strict check rejected), with a
      regression test.
- [x] Helper scripts ported to macOS and Linux as well as Windows Git Bash. Only
      `phase0-verify.sh` has been rerun; see STATUS.md.
- [x] LICENSE file added (Apache-2.0, as `Cargo.toml` and the README already said).
- [ ] Confirm CI is green on the pull request that carries this work, including the
      private-Tor test, whose script changed with the protocol.

## 1. Finish the chain

The network layer does everything SPEC sections 7.1, 8 and 13 ask of it, on one machine
and in CI.

- [ ] **1.1 Decide the inference core.** Compare the in-repo runtime (llama.cpp
      layer-range patch, `crates/dfin-llama*`, `crates/dfin/src/stage*`) with existing
      split-inference projects, starting from `docs/prior-art-2026-11.md` (Mesh LLM /
      skippy is the closest living one). Judge on: exact output against the unsplit
      model, support for `glm-dsa`, `glm5-next` and `kimi-k3`, wide stage state, sharded
      loading, licence, how much of dfin's Tor transport and sealed routes would survive,
      and maintenance cost.
      **Done when:** a written recommendation (keep, adopt, or converge) with evidence is
      in `docs/` and the owner has chosen. No code is swapped by this task.
      **Needs:** nothing. Do this first: tasks 1.3 and 1.4 touch the stage engine.
- [ ] **1.2 Compartmentalisation cap** (SPEC 7.1). The client refuses a route on which
      one node key holds more than the cap or on which fewer than two keys serve the
      swarm's layers; the hub refuses an announcement above the cap; `/status` shows each
      node's share; development mode may ignore it. Propose the default cap in the pull
      request (SPEC open question 2).
      **Done when:** route-choice tests cover the cap, an end-to-end test shows a
      refused single-stage route in Tor mode, and protocol.md describes it.
- [ ] **1.3 Opt-in conversation sessions** (SPEC 8.1). An explicit, per-conversation
      opt-in on the local API keeps one client key and route so the prefix cache
      survives between requests. Default behaviour (fresh identity per request, nothing
      cached) is unchanged. The daemon discloses what a session links and reports it in
      `/status`; sessions end on request, on idle timeout and on shutdown, and stages
      then drop their state.
      **Done when:** an end-to-end test over the Tor fixture shows a prefix-cache hit on
      the second request of a session; the existing unlinkability regressions still pass
      for requests without one; protocol.md describes the opt-in.
- [ ] **1.4 Failover of a live session** (SPEC goal 7, v0.1 section 7). A per-session
      replay log on each stage, the client rebuilding the route and replaying into the
      replacement. Only for opted-in sessions.
      **Done when:** an end-to-end test kills a stage mid-generation and the stream
      resumes with identical output after a pause, with no error to the caller.
      **Needs:** 1.3.
- [ ] **1.5 Encrypt activations end to end across the application relay** (SPEC 8.2), so
      a `dfin relay` carries ciphertext. If that proves not worth its cost, bring the
      owner the alternative of removing the relay return mode.
      **Done when:** a relay privacy test shows the relay cannot decode a model-state
      frame, and `relay_can_read_model_states` is gone from `/status`.
- [ ] **1.6 Locked weights by default** (SPEC 13). A stage locks its mapped weights at
      load and does not announce or serve until the lock completes; opting out is
      explicit.
      **Done when:** covered by a stage test, and the cold-start limitation is removed
      from STATUS.md.
- [ ] **1.7 Tool calls in the OpenAI layer.** Parse tool calls per chat template; handle
      chat templates of thinking models. Today `tools` is accepted and ignored.
      **Done when:** an end-to-end test returns a well-formed `tool_calls` response for
      at least the GLM template.
- [ ] **1.8 Find the intermittent handshake failure.** CI run 42404522275 failed once in
      `stage::privacy_tests::repeated_disconnects_release_client_identities_and_engine_handles`
      with a peer-aborted handshake; the cause was never found and later runs passed.
      It failed again once on macOS on 9 November 2026 during a full-suite run
      ("handshake timed out"), then passed 12 of 12 alone. Add a protocol version check
      to hub and relay connections while in this code.
      **Done when:** the cause is identified and fixed, or the test has run 500 times in
      a loop on Linux without failing and that is recorded in STATUS.md.
- [ ] **1.9 Widen CI.** Add a macOS job, and `cargo clippy` and `cargo fmt --check`
      gates (format the workspace once, in its own commit, first).
      **Done when:** all of them run on every pull request.

## 2. Prove it on the public Tor network

The first real measurement of what the spec promises: stages as onion services on
separate machines, a real model, the public Tor network.

- [ ] **2.1 Operator runbook.** One page, tested from a clean machine on each OS the
      owner has: install Tor, publish an onion service, start a stage, start a client.
      Supersedes the setup half of `docs/anonymity.md`.
- [ ] **2.2 Measure.** GLM-4.7-Flash across at least two stage machines plus the client,
      a route that satisfies the cap from 1.2, with and without a session: time to first
      token, tokens per second, per-hop time, recovery after killing a stage.
      **Done when:** the numbers and the exact commands are in STATUS.md and a dated log
      is in `docs/logs/`; SPEC acceptance test 2 is demonstrated.
      **Needs:** 1.2, 1.3; the owner's machines, with the owner running the commands on
      the ones an agent cannot reach.

An earlier attempt (one Mac client, one Linux stage holding 45 of 47 layers) was
prepared on 7 November and abandoned before it ran. That layout would now be refused by
the cap.

## 3. Tamper detection on consumer nodes

SPEC section 7.2 and the open and pod tiers of section 5. Everything here runs on the
owner's machines and in CI.

- [ ] **3.1 Sanity checks on every received activation**: finiteness and norm bounds,
      at each stage and at the client tail. None exist today.
- [ ] **3.2 A misbehaving stage for tests**: a test-only stage that adds a steering
      vector, returns garbage, or lags, so every later task has something to catch.
- [ ] **3.3 Redundant execution.** Fork a stage's input to a second independent node,
      compare with a cosine threshold, attribute with a third. Sampled at 1 to 2 percent
      by default; every token when the request has `tools`.
- [ ] **3.4 The threshold.** Measure output differences between honest backends (Metal,
      Vulkan, CUDA, CPU) on a real model and set the threshold from data (SPEC open
      question 1).
- [ ] **3.5 Shadow admission** for new nodes, and **reputation** on long-lived node keys
      in the directory, with a delisting rule.
- [ ] **3.6 Signed output logs** per session, auditable by replay.
- [ ] **3.7 Trust tiers in the API**: `:open` (default) and `:pod` with vouching; the
      attested suffixes fail with a clear message; `/status` reports tier, shares and
      how many tokens were checked; the one-time disclosure for open routes.
- [ ] **Milestone done when:** SPEC acceptance test 4 passes in CI: the misbehaving
      stage from 3.2 is caught and delisted within 1,000 tokens, and with `tools`
      present no steered token reaches the caller.

## 4. Money

SPEC section 10. Monero stagenet only until the owner says otherwise.

- [ ] **4.1 Design pass, reviewed by the owner before any code.** Exactly what the mint
      and each node can correlate, how served tokens are metered and proven, how payouts
      are computed, refunds, and timing and amount correlation. Assess the Cashu
      Development Kit with an XMR backend against that.
- [ ] **4.2 Mint**: blind credits against XMR deposits, redemption, full-reserve
      accounting with Monero reserve proofs.
- [ ] **4.3 Wallet in the client daemon**; balance in `/status`.
- [ ] **4.4 Per-token payment along the chain** with blinded change; payment only on
      delivery.
- [ ] **4.5 Operator payouts** per epoch; bonds posted and forfeited (ties to 3.5);
      balance caps.
- [ ] **Milestone done when:** on stagenet, a stranger's node is paid for served tokens,
      the reserve proof matches credits outstanding, and a written check shows no party
      can join a purchase to a usage record (SPEC acceptance test 6).
- [ ] **Owner gate before real funds:** legal review of money transmission, payout tax
      reporting and the model licences.

## 5. No centre

SPEC section 12 and goal 6.

- [ ] **5.1 Several hubs**: peering, mirroring, and a client that compares independent
      directory views and notices a personalised one.
- [ ] **5.2 Bootstrap without DNS**: onion seed lists compiled into the binary.
- [ ] **5.3 Operator independence**: research what evidence a directory can carry
      without identifying operators (SPEC open question 3), then build it.
- [ ] **5.4 Mint federation**: trust lines between mints, settlement in XMR.
- [ ] **5.5 Reproducible builds and signed releases**: pinned toolchain, container
      build, published hashes, several signing keys, a mirror on a second host.
- [ ] **5.6 Tor inside the binary** (arti), so a user installs one thing, if SPEC open
      question 5 resolves in its favour.
- [ ] **5.7 Chaff sessions and the privacy-depth option** (SPEC 6.2).
- [ ] **Milestone done when:** SPEC acceptance test 7 passes: two hubs and two mints
      peer, the network serves with the project's own hub and mint off, and a third
      party reproduces the release byte for byte.

## 6. Frontier scale

SPEC section 13 and acceptance test 8. The measurement in 6.2 is the go or no-go for
frontier-size models. It is independent of milestones 3 to 5, so run it as soon as the
owner agrees to rent the hardware rather than waiting for them.

- [ ] **6.1 Sharded loading of GLM-5.3** (about 365 GB) and confirmation that mainline
      llama.cpp runs it at all (`docs/prior-art-2026-11.md` flags this as unverified).
      **Needs:** rented machines.
- [ ] **6.2 The scale measurement**: GLM-5.3 at 3 tokens per second or better across 4
      to 8 machines; cold 20k-token prefill under 30 s. **Needs:** rented machines.
- [ ] **6.3 Micro-batching across sessions** in the stage engine.
- [ ] **6.4 Speculative decoding**: the MTP head in the client tail and a client-side
      draft model; measure the acceptance rate (SPEC open question 4).
- [ ] **6.5 Kimi K3**: boundary rules and state snapshots for KDA layers, anchor pods,
      and a tail that fits a 16 GB client (SPEC open question 6).
- [ ] **6.6 FP8 activations** between stages.
- [ ] **6.7 Gateway**: the head and tail as a hosted onion service for users who will
      not run the daemon, with its trust stated plainly.
- [ ] **6.8 An agent finishes real work**: OpenCode completes a coding task through the
      network.

## 7. Attested tier (optional add-on)

SPEC sections 5 and 6.3. Nothing above may depend on this.

- [ ] **7.1** Reproducible confidential-VM stage image (SEV-SNP or TDX) with a published
      measurement.
- [ ] **7.2** Attestation verification in the client before any activation is sent; the
      `:attested` and `:ends` tiers; attestation evidence in the directory.
- [ ] **Milestone done when:** someone outside the project verifies an attested route
      end to end (SPEC acceptance test 9). **Needs:** rented enclave servers.
