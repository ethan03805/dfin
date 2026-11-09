# Inference core: keep, adopt or converge

Evaluation for [PLAN.md](PLAN.md) task 1.1, written 9 November 2026. It compares dfin's
own split-inference core with existing projects and ends in a recommendation. **The
decision is the owner's** (AGENTS.md rule 9); task 1.1 stays unticked until it is made.
No code was changed for this evaluation.

Everything said here about another project was read from its source tree, its issue
tracker or its release list on the date above, and the claims link to where they were
read. Mesh LLM was read at commit
[`213792f`](https://github.com/Mesh-LLM/mesh-llm/tree/213792f5abb6dd0b19e313a282c4636b9a3bce6c)
(9 November 2026). Nothing from any candidate was built or run; see
[What could not be verified](#what-could-not-be-verified).

## Recommendation

**Keep the in-repo core. Converge with Mesh LLM's skippy on three specific pieces. Do
not adopt it, at any level, today.**

1. **Keep** the llama.cpp layer-range patch, `crates/dfin-llama*` and the stage engine.
   It is the only core found that runs all three target architectures as stages with
   their wide state, though so far only on tiny synthetic models, and it is small: one
   patch of 1,376 lines against a 65,904-line patch queue.
2. **Adopting skippy would not reduce what dfin maintains.** Every part of dfin that
   makes it dfin (Tor transport, sealed routes, per-request identities, the fuzzed
   decoders, the client holding the head and tail) sits above the inference core and
   would have to be kept whichever core is underneath. What would change is that the
   1,376-line patch is replaced by a dependency on a much larger patch queue behind an
   interface its own maintainers label experimental and are considering replacing.
3. **Skippy does not meet the architecture requirement today.** Of `glm-dsa`,
   `glm5-next` and `kimi-k3`, only `kimi-k3` is on its split-certification roster. Its
   tracker has an open report that GLM-5.3-Flash does not split, and its own documents
   say `glm-dsa` is not offered as a runnable stage family.
4. **Converge on:** its way of testing splits (every cut, real weights, a roster that
   refuses uncertified architectures); its per-layer package layout, so that dfin can
   read packages that already exist; and its graph-derived way of finding what crosses
   a stage boundary, as the design to move to if hand-wiring architectures becomes a
   burden. Details in [What to converge on](#what-to-converge-on).
5. **Look again when** skippy certifies `glm-dsa` and `glm5-next` on real weights and
   its native interface settles, or when dfin starts PLAN milestone 6 (micro-batching,
   speculative decoding, recurrent-state snapshots), where skippy has working code and
   dfin has none. At that point the option to price is skippy's engine layer alone
   (its patched llama.cpp and `skippy-ffi`), never its serving layer or Mesh.

The reasoning is in the sections below; the risks of this recommendation are in
[Risks](#risks).

## What was compared

**The in-repo core.** Mainline llama.cpp at commit `45b6604b` (5 November 2026) plus
`vendor/patches/0001-layer-range.patch`; the bindings and safe wrapper in
`crates/dfin-llama-sys` and `crates/dfin-llama`; the stage engine in
`crates/dfin/src/stage/engine.rs`. The patch adds `layer_begin` and `layer_end` to the
model parameters, makes the loader skip tensors and absent shard files outside the
range, and makes each wired graph builder run only its layers, taking the hidden state
through `llama_batch.embd` and returning the boundary state as embeddings.

**Mesh LLM and skippy.** One repository,
[Mesh-LLM/mesh-llm](https://github.com/Mesh-LLM/mesh-llm), holding two products
([ARCHITECTURE.md][sk-arch]):

- **skippy**: model management, the staged runtime, a stage wire protocol and an
  OpenAI-compatible frontend, on a patched llama.cpp. It does not depend on Mesh.
  Inside it there are two layers worth telling apart: the **engine** (the patch queue,
  the C interface it adds, `skippy-ffi`, `skippy-runtime`) and the **serving layer**
  (`skippy-serving`, `skippy-protocol`, the topology planner).
- **Mesh**: peer discovery, identity, routing and transport around skippy.

They are evaluated as three possible adoptions: the engine alone, skippy with its
serving layer, and the whole of Mesh LLM.

**Others.** The remaining projects in [prior-art-2026-11.md](prior-art-2026-11.md)
were rechecked and a search was made for anything newer; it found five more. None is
a serious alternative, so they are covered briefly in
[Other candidates](#other-candidates).

## Comparison

| | dfin core (keep) | skippy engine only | skippy serving layer | Mesh LLM whole |
|---|---|---|---|---|
| **1. Split equals unsplit** | Tested: tiny models in the suite match within 1e-3 on every logit with the same argmax; GLM-4.7-Flash gives identical text at f32 | Tested more widely: per-architecture certification on real weights (same next token), structural tests of every cut, native fixtures comparing logits and state | Same engine | Same engine |
| **2. `glm-dsa`** | Wired; tiny model only, by script, not in the suite | Annotated, **not certified**; documents say it is not a runnable stage family | Refused at startup unless overridden | Same |
| **2. `glm5-next`** | Wired with its four streams; tiny model in the suite | Support patch dated 6 November 2026, **no certification run**; open issue that the real model does not split | Refused at startup unless overridden | Same |
| **2. `kimi-k3`** | Wired with its residual bank; tiny model in the suite | **Certified** on a 0.40B, 8-layer model | Admitted | Same |
| **2. Wide stage state** | Per-architecture hook, hand-written for two architectures | Derived from the graph for any annotated architecture | Carried as typed multi-part frames | Same |
| **3. Only its own layers, per-layer shards** | Yes; loader tolerates absent shards | Yes; per-layer fragments assembled into a stage file | Yes | Yes, fetched from Hugging Face by default |
| **4. Licence** | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| **5. dfin transport, sealed routes, identities, fuzzers survive** | All | All; only `dfin-llama*` and the engine thread are replaced | **No**: its protocol replaces `dfin-proto` and the chain, and its data flow gives nodes token ids | **No**: iroh QUIC with relays, Nostr discovery, usage analytics on by default |
| **5. Conflicts with Tor-only** | None | None found; the engine has no network code | Plain TCP between stages, a coordinator-chosen topology, a direct return connection | Direct UDP paths preferred, public relays, no Tor support anywhere in the tree |
| **6. Patch carried on llama.cpp** | 1 patch, 1,376 lines, 26 files | 113 patches, 65,904 lines | Same | Same |
| **6. Code above the patch** | About 1,300 lines (wrapper and engine thread) plus generated bindings | About 43,000 lines (`skippy-ffi`, `skippy-runtime`), an interface versioned 0.1.x | About 370,000 lines of Rust in skippy | Plus about 567,000 lines in Mesh |
| **6. Tracking upstream** | By hand; has been pinned once and never rebased | Re-pinned on 19 October and 9 November 2026 with an automated certification gate | Same | Same |
| **7. Health** | One committer | Very active: about 3,100 commits since February 2026, three contributors with over 400 commits each | Same | Same |
| **Verdict** | **Keep** | Not today; the one to revisit | Not compatible with SPEC 6.2 and 8.2 | Not compatible with SPEC 0 |

## Evidence

### 1. Is split output equal to unsplit output, and how is that tested?

**dfin.** `crates/dfin-llama/tests/tiny_model.rs` and `tests/wide_state.rs` run a tiny
model whole and as three pieces and require the largest absolute logit difference to be
under 1e-3 and the argmax to agree at every position, for prefill, a decode step and a
rollback. The tolerance exists because assembling state in pieces reorders float
additions; so the suite shows agreement to that tolerance, not bit-identical logits.
The suite covers tiny `llama`, `kimi-k3` and `glm5-next` models. `glm-dsa` is covered
only by `examples/layer-range` inside the patched checkout, run by
`scripts/phase0-verify.sh` (`docs/logs/phase0-verify-2026-11-05.log`). On a real model
the evidence is one model: GLM-4.7-Flash produced identical text split and unsplit at
f32 activations (STATUS.md, "Measured"); at f16 near-tie tokens flip. One three-piece
split is tested per model, not every boundary. All of these tests skip silently when
the tiny models are absent.

**skippy.** Three layers of testing:

- A family is "certified" when a two-stage split and the recommended multi-stage split
  each produce the same next token as the whole model on the recommended real artifact
  ([FAMILY_CERTIFY.md][sk-certify]; the comparison in
  [`single_step.rs`][sk-single] checks the first and second predicted token). This is
  token equality on real weights, which is weaker per position than a logit bound but
  is run on real models.
- Synthetic models for every registered architecture check every interior cut and a
  three-stage chain structurally, and "do not run numerical inference"
  ([SYNTHETIC_GRAPH_CONTRACTS.md][sk-synth]). The same page describes separate native
  fixtures that compare logits and serialized sequence state against independently
  loaded contexts, covering split prefill and decode, recurrent state and MTP.
- Split startup **fails closed** unless the model's architecture is on a roster bound
  to the llama.cpp pin, the native interface version and the patch-queue hash
  ([SKIPPY_SPLITS.md][sk-splits], [`split-certified.json`][sk-roster]: 84
  architectures at this commit).

Activation frames are raw f32; lower-precision wire formats were tried and are recorded
as research only ([llama-parity-candidates.json][sk-cands], `activation_transport_note`),
which matches dfin's own f16 finding.

**Judgement.** Both test the property. Skippy's regime is broader and stricter about
admission; dfin's is narrower but includes the two architectures skippy has not
certified. Neither proves bit-identical logits.

### 2. The three architectures and wide stage state

**dfin.** The patch wires eight graph builders: `llama`, `qwen3`, `qwen3moe`,
`deepseek2`, `glm4-moe`, `glm-dsa`, `glm5-next` and `kimi-k3`. A plain residual
architecture costs six to ten changed lines; `glm5-next` costs 36 and `kimi-k3` 35,
because each also implements `stage_state_mult(il)` and splits a wide input back into
streams, or bank plus prefix sum ([protocol.md](protocol.md) section 11). `glm-dsa`
stages must start at a full-indexer layer, and the patch enforces it. None of the three
has been run on real weights: GLM-5.3 has never been loaded (STATUS.md), and an
upstream bug about a GLM-5.3-Flash quant crashing at load was open on the day of
writing ([llama.cpp #31947](https://github.com/ggml-org/llama.cpp/issues/31947)).

**skippy.**

- `kimi-k3` is on the roster, certified on
  `murillo2000/Kimi-K3-0.40B-GGUF` (8 trunk layers, width 1024), described as a
  "compact hybrid three-recurrent-to-one-attention boundary sweep"
  ([`family-certified.json`][sk-famcert]). The synthetic suite states that its Kimi
  fixture "preserves cross-layer residual checkpoints". The full-size model is not the
  certified artifact.
- `glm5-next` is not on the roster. An issue opened 28 October 2026 and still open
  reports that the GLM-5.3-Flash layer package cannot be split: after fixing a stale
  architecture name, "the native planner rejects the graph" because the KDA recurrent
  state and the indexer writes are not registered as stage state
  ([mesh-llm #2663](https://github.com/Mesh-LLM/mesh-llm/issues/2663)). A support
  patch dated 6 November 2026 has since been added
  ([`0011-models-expose-glm5-next-stage-graph-semantics.patch`][sk-glm5]), and the
  candidate list records the family as "no local artifact certification run yet"
  ([llama-parity-candidates.json][sk-cands]).
- `glm-dsa` is not on the roster. A generated two-line annotation patch exists for it,
  but the runtime contract says the pinned runtime "does not currently carry the
  GLM-DSA staged graph patch" and that "certification and serving must fail closed
  until the native runtime support is restored"
  ([GLM_DSA_RUNTIME_CONTRACT.md][sk-dsa]). Closed issues show GLM-DSA work in an
  earlier queue ([#1076](https://github.com/Mesh-LLM/mesh-llm/issues/1076),
  [#1231](https://github.com/Mesh-LLM/mesh-llm/issues/1231)). Whether the annotation
  or the contract describes the current tree could not be settled by reading alone.
- **Wide state** is where skippy's design is ahead. Its core patches slice the live
  ggml graph between `begin_block` and `end_block` markers and report every value that
  crosses the cut, so the boundary's width falls out of the graph rather than out of
  hand-written code per architecture; each further architecture is a generated
  annotation of a few lines. Frames carry "a versioned multipart directory with identity,
  dtype, dimensions, strides, and a token axis for every graph-frontier value"
  ([skippy-protocol README][sk-proto]). The synthetic registry shows it working for an
  architecture with four hyper-connection streams (`qwen4exp`, boundary width four
  times the embedding width).

**Judgement.** On the day of writing, dfin's core is the only one of the two in which
all three target architectures run as stages, and that is shown on tiny synthetic
models only. Skippy has one of three certified and the other two in progress.

### 3. Can a node load only its own layers from per-layer shards?

Both can. dfin: `tools/shard_gguf.py` writes metadata, token embedding, one file per
layer and the output head; the patched loader skips absent files
([protocol.md](protocol.md) section 10; tested in `crates/dfin-llama/tests/shards.rs`
and end to end). skippy: a layer package is `shared/metadata.gguf`,
`shared/embeddings.gguf`, `shared/output.gguf` and `layers/layer-NNNNN.gguf` with
sizes and SHA-256 in a manifest; "each node resolves the package and downloads only
the shared artifacts plus the layer files needed for its assigned stage", then
assembles a stage file in a local cache ([LAYER_PACKAGE_REPOS.md][sk-pkg],
[SKIPPY_SPLITS.md][sk-splits]). The two layouts are nearly the same shape. Skippy
fetches from Hugging Face by default; fetching between peers is off by default on
public meshes.

### 4. Licence

Mesh LLM, including skippy, is Apache-2.0
([LICENSE](https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/LICENSE);
GitHub reports the same). dfin is Apache-2.0 and llama.cpp is MIT, so code can move in
either direction with attribution. No licence obstacle.

### 5. How much of dfin survives, and does the candidate assume its own network?

**Engine only.** `skippy-ffi` and `skippy-runtime` have no network dependency
(`skippy-runtime` depends on tokio for `sync` only), and no skippy crate depends on
iroh. An adoption at this level would replace `crates/dfin-llama*` and the engine
thread in `crates/dfin/src/stage/engine.rs` and leave `dfin-net`, `dfin-proto`,
`dfin-fuzz`, the client, the hub and the relay as they are. The native interface is
"experimental version-0" ([ARCHITECTURE.md][sk-arch]); it went from 0.1.74 to 0.1.79
between 28 October and 9 November 2026 (issue #2663, roster file), and an open
tracking issue considers replacing it altogether
([mesh-llm #2726](https://github.com/Mesh-LLM/mesh-llm/issues/2726)).

**Serving layer.** Not compatible with the spec, for reasons that are structural:

- Stage 0 receives token ids and the final stage produces logits and returns the
  predicted token ([DATA_FLOW.md][sk-flow]). SPEC 6.2 requires that no node sees token
  ids or logits. Skippy's lock file also requires that each node be assigned once
  ([SKIPPY_SPLITS.md][sk-splits]), so the client could not be both ends of a chain.
- A coordinator chooses the topology; in dfin the client builds the route and each hop
  sees only its sealed record (SPEC 8.2).
- The final stage opens a connection straight back to stage 0 for predictions, and
  activation traffic runs over plain TCP between neighbouring stage servers
  ([`binary_messaging.rs`][sk-tcp]), with authentication and encryption left to Mesh.
- The stage protocol is "internal, generation-gated ... mixed generations fail closed;
  it is not a backwards-compatible public network API" ([ARCHITECTURE.md][sk-arch]),
  at generation 16 on the day of writing. No fuzz target for the stage wire protocol
  was found in the tree, which AGENTS.md rule 3 requires of every dfin wire type; the
  fuzz targets present belong to a separate C++ library for embedding and reranking.

**Mesh.** Transport is QUIC through iroh, which chooses between direct UDP paths and
relays; discovery is Nostr relays or mDNS; join is by invite token
([DESIGN.md][mesh-design]). Their split guidance says to "prefer directly reachable,
low-latency UDP paths". A search of the tree for Tor, onion or SOCKS support found
none. Mesh ships opt-out usage analytics
([mesh-llm-analytics README][mesh-analytics]). This is a different product with a
different threat model, and its network layer contradicts SPEC section 0 ("Tor only")
outright.

### 6. Maintenance cost either way

| | dfin | skippy |
|---|---|---|
| Patch on llama.cpp | 1 patch; 1,376 lines; 26 files; +700 −43 | 113 patches; 65,904 lines; +45,516 −1,502. 16 core patches (47,365 lines) add a staging subsystem under `src/skippy`; 81 generated annotations (4,219 lines); 16 model-support patches (14,320 lines) |
| Per new architecture | 6 to 36 hand-written lines in its builder; more if its state is wide | A generated annotation of a few lines, then certification |
| llama.cpp pin | `45b6604b`, 5 November 2026 | `372dc5ca`, 8 November 2026, 43 commits ahead of dfin's |
| Pin moves | None since the first | 19 October and 9 November 2026; the pin file's history at its present path starts on 5 October |
| Who does the rebase | dfin | The Mesh LLM maintainers, with a CI "canary" that regenerates the roster and blocks a pin that does not pass |
| Rust above the patch | `dfin-llama` 892 lines, engine 411, generated bindings 3,689 | `skippy-ffi` about 8,500, `skippy-runtime` about 35,000 |
| Releases | None | Stable releases every one to three weeks through `v0.78.1` (5 October 2026), a preview on 30 October |

If dfin adopted the engine, it would stop rebasing its own patch and start following
someone else's pin, interface version and patch queue, with no say in when the
architectures dfin needs are certified. If dfin keeps its core, it carries a small
patch whose risky parts are the two wide-state builders, which upstream may refactor.
dfin has never rebased the patch, so the real cost of a pin move is not known; see
below.

One thing skippy has that dfin will need: working code for recurrent-state rollback
and snapshots on KDA layers, state export and import, multi-lane batching and
speculative decoding with the model's own MTP head. These are PLAN tasks 1.3, 1.4 and
most of milestone 6. This is the strongest argument for adoption and the reason for
the revisit condition in the recommendation.

### 7. Project health

Mesh LLM: created 11 February 2026; last push 9 November 2026; about 3,100 commits on
`main`; 4,004 stars and 505 forks; top contributors by commits 1,154, 834, 442 and
153; 132 open issues and 32 open pull requests (GitHub API, 9 November 2026). Open
issues that bear on correctness of splits: #2663 (GLM-5.3-Flash does not split), #2318
(speculative-verify overheads on hybrid models), #1864 (auxiliary certification for
three families). Their own split guide records "concurrent admission and suffix cache
reuse as active validation gaps" ([SKIPPY_SPLITS.md][sk-splits]). Their support matrix
page was last updated 28 August 2026 and lags the roster. The project is healthy and
fast; the speed is also the cost, since its internal interfaces change weekly.

dfin: one committer on `main`, 124 tests, one known intermittent failure (STATUS.md).

## Other candidates

The projects in [prior-art-2026-11.md](prior-art-2026-11.md) were rechecked and a
search was made for newer ones. It found five that the earlier survey missed. None
meets dfin's requirements, and none changes the recommendation. Dates and licences are
from the GitHub API on 9 November 2026. Rows marked † had their main claims checked a
second time against the repository's README or documents; finer details, such as
which architectures a patch wires, and the unmarked rows come from a single survey
pass.

**Built on llama.cpp**, so able in principle to use its `glm-dsa`, `glm5-next` and
`kimi-k3` builders:

| Project | Last push | Licence | What it is | For dfin |
|---|---|---|---|---|
| llama.cpp itself † | Mainline | MIT | No way to run layers `[a, b)`. A request for exactly that, "run layers [il_start, il_end) with an injectable/extractable boundary resid[ual]", was closed as not planned on 6 October 2026 with no maintainer reply ([#25577](https://github.com/ggml-org/llama.cpp/issues/25577)). The upstream direction is the RPC backend: [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626) (asynchronous backend APIs) merged 26 August 2026, and server-to-server tensor copies ([PR #8032](https://github.com/ggml-org/llama.cpp/pull/8032)) are still open, so one host holds the graph and every boundary crosses it | Nothing to adopt. dfin should expect to carry its patch indefinitely |
| [Unsloth PR #180](https://github.com/unslothai/llama.cpp/pull/180) † | Closed unmerged 4 September 2026 | MIT | "Layer-split pipeline parallel: run a contiguous layer range per node, bit-exact". Wired `qwen35` and `qwen3moe`; its author closed it in favour of RPC | Abandoned. Already the model for dfin's patch |
| [Hydra](https://github.com/PavanManchikatla/Hydra) † (new) | 25 September 2026 | Has a LICENSE file GitHub does not classify; not read | Rust coordinator over a llama.cpp fork with a layer-window patch, per-stage shard files with a signed manifest, mutual TLS on a trusted LAN, fuzzed parsers. Its patch covers `llama` and `qwen2` | The same design as dfin with fewer architectures, a trusted-LAN threat model and no users. Nothing to gain |
| [prima.cpp](https://github.com/OpenCPIL/prima.cpp) | 6 September 2026 | MIT | A 2024 fork of llama.cpp; ring of layer windows over ZeroMQ; every device needs the whole file. Now at `OpenCPIL/prima.cpp`, not gone from GitHub as the earlier survey said | Too old a fork to have the target architectures |

**Other engines that cover some of the target models:**

| Project | Last push | Licence | What it is | For dfin |
|---|---|---|---|---|
| [DwarfStar](https://github.com/antirez/ds4) † (new) | 8 November 2026 | MIT | Its own C engine for a narrow set of models, including GLM 5.2, GLM 5.3 and GLM 5.3 Flash. Has a pipeline mode: `--layers a:b`, "each process maps only its assigned layers". Reads only the GGUF files the project itself produces ("not a general GGUF runner"). "Network protocols have no authentication or encryption" | Not adoptable: no Kimi K3, its own file format, a second engine to bind. Worth knowing as an independent implementation of the GLM models to compare real-weight output against |
| [Shard](https://github.com/leyten/shard) † (new) | 28 October 2026 | Apache-2.0 | One PyTorch engine per model, among them Kimi-K3; a run of GLM-5.2 split thirteen layers per node over the public internet; claims output "bit-identical" to one machine, with signed receipts that record GPU identifiers and public IP addresses | Not GGUF, not consumer CPU or Metal, and its receipts publish what dfin exists to hide. The only project found with a Kimi-K3 split engine |
| [SwarmLLM](https://github.com/enapt/SwarmLLM) † (new) | 9 November 2026 | MIT or Apache-2.0 | Own Rust engine on Candle that reads GGUF; peers hold parts of a model; libp2p with hole punching. Its README tells users a node could reconstruct their question | Closest in product shape; none of the three architectures; its own network |
| [Parallax](https://github.com/GradientHQ/parallax) † (new) | 1 July 2026 | Apache-2.0 | Pipeline sharding over SGLang or vLLM on GPUs and MLX LM on Macs; its own peer-to-peer layer | Not llama.cpp, not GGUF |
| [exo](https://github.com/exo-explore/exo) † | 6 November 2026 | Apache-2.0 | MLX; tensor parallelism | A different kind of split |

**The rest**, unchanged in substance from the earlier survey:
[distributed-llama](https://github.com/b4rtaz/distributed-llama) (MIT, active, tensor
parallel, own format); [cake](https://github.com/evilsocket/cake) (FAIR licence, free
only for non-commercial use, so not usable); [dnet](https://github.com/firstbatchxyz/dnet)
(Apache-2.0, MLX only); BloomBee and
[Petals](https://github.com/bigscience-workshop/petals) (PyTorch; Petals dormant since
2024); [Kalavai](https://github.com/kalavai-net/kalavai-client), GPUStack and LocalAI
(orchestration of whole-model engines or of llama.cpp's RPC workers, not split
engines).

Two things follow. First, the layer-window idea has now been built on llama.cpp at
least four times (Unsloth, Hydra, skippy, dfin) and upstream has declined it, so none
of these projects can expect to drop its patch. Second, nobody else combines GGUF
layer ranges with all three target architectures; skippy is the nearest, and
DwarfStar and Shard each cover one side with a different engine.

## What to converge on

1. **Testing.** Adopt skippy's three ideas in dfin's own suite: test every interior
   boundary rather than one three-piece split; add a next-token comparison on real
   weights for each target architecture as soon as weights can be loaded; and make a
   stage refuse an architecture that has not passed, rather than run it. The last one
   closes a gap found in passing (below).
2. **Shard layout.** dfin's shard set and skippy's layer package hold the same pieces.
   A reader or converter for skippy packages would let dfin use packages that already
   exist, subject to the licence review in SPEC section 17 and to dfin's rule that
   weights move between onion peers. Whether a converter is as simple as it looks was
   not tested.
3. **Graph-derived boundaries.** If the list of architectures dfin must serve grows,
   or upstream refactors the two wide-state builders, move from hand-written
   `stage_state_mult` code to marking block boundaries and deriving what crosses them
   from the graph, as skippy's core patches do. This is a design to borrow, not code
   to copy wholesale: their implementation is the 47,365-line part of their queue.

## Risks

**Of keeping the core**

- **Real weights are unproven for all three targets.** Everything dfin knows about
  `glm-dsa`, `glm5-next` and `kimi-k3` as stages comes from tiny synthetic models. A
  failure on real GLM-5.3 would be found at PLAN task 6.1, late.
- **One committer and one patch.** A refactor of `glm5-next.cpp` or `kimi-k3.cpp`
  upstream lands on dfin alone. The patch has never been rebased, so the cost is
  unmeasured.
- **Features still to build.** Recurrent-state snapshots, session replay,
  micro-batching and speculative decoding all touch the engine and all exist in skippy
  today. Keeping the core means building them.
- **Eight architectures, not all.** See the first follow-up below.

**Of adopting skippy's engine**

- It cannot serve two of the three target architectures today, and dfin would not
  control when it can.
- Its native interface is experimental, moved from 0.1.74 to 0.1.79 in twelve days,
  and may be replaced (#2726).
- dfin would inherit a 65,904-line patch queue it did not write and could not
  realistically audit, in a project where "every byte from a peer is hostile".
- Its certification and rebase process leans on its own CI and hardware; dfin would be
  trusting results it cannot reproduce.

**Of converging**

- Reading skippy packages ties a dfin tool to a manifest format that rejects unknown
  fields and changes by schema version.

## What could not be verified

- **Nothing from Mesh LLM was built or run.** Every statement about skippy's behaviour
  comes from its documents, tests, patches and issues. In particular its certification
  results were read, not reproduced.
- **Whether skippy runs `glm-dsa` as a stage today.** A generated annotation patch
  exists and synthetic GLM-DSA cases are described as passing in a design note, while
  the runtime contract and candidate list say the family is not runnable. Only a build
  would settle it.
- **Whether the 6 November `glm5-next` patch fixes issue #2663.** The issue is still
  open and no certification run is recorded.
- **Whether skippy's engine can run a head piece and a tail piece in one process**, as
  dfin's client does. The C interface opens a model from parts for a layer range and
  looks able to; the serving layer does not allow it.
- **That the Hugging Face layer packages named in skippy's documents and issues exist
  and load** (`meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers` and others). They were not
  fetched.
- **The cost of rebasing dfin's patch** onto a newer llama.cpp. It has not been done
  since the first pin.
- **What dfin's patch does when a layer range is requested for an architecture it has
  not wired.** No explicit rejection was found by reading the patch; it was not run.
- **dfin's own test results.** The suite was read, not rerun, for this evaluation;
  STATUS.md is the source for what passes.
- **The other candidates beyond their READMEs, documents and repository metadata.**
  None was built or run. Not checked: whether DwarfStar's pipeline mode has been
  exercised with GLM 5.3 Flash (its examples use DeepSeek models); Shard's
  bit-identical claim, its receipts and the licence of the Kimi reference code it
  vendors; Hydra's licence text; exactness tests in any project other than skippy,
  Unsloth PR #180 and Hydra; BloomBee's current repository; the Gitee copy of
  prima.cpp.
- **That the search for newer projects is complete.** It was a set of GitHub and web
  searches, and GitHub code search was rate-limited partway through.

## Found in passing, not acted on

- **The patch wires 8 architectures; llama.cpp has 174 model source files.**
  `prior-art-2026-11.md` says "every architecture llama.cpp supports then works as a
  stage", and SPEC section 15 states the requirement as "any llama.cpp architecture
  works as a stage". Neither is true of the patch as it stands. The owner should decide
  whether the requirement is the three target architectures (and the sentences are
  corrected) or all of them (and the patch needs the graph-derived approach above).
- **No guard for unwired architectures** (see above). A stage should refuse, not run
  all layers on a partial load.
- **`prior-art-2026-11.md` needs correcting.** On Mesh LLM: the pin is now 8 November
  2026, the queue is 16 core, 81 generated and 16 model-support patches, and its
  design has moved from per-family patches to graph-derived staging. Elsewhere:
  prima.cpp is on GitHub at `OpenCPIL/prima.cpp`; llama.cpp PR #18626 merged in
  August and does not add layer-range execution; and DwarfStar, Shard, Hydra,
  SwarmLLM and Parallax are missing from it.
- **Upstream llama.cpp issue #31947** (GLM-5.3-Flash crash at load) is worth reading
  before PLAN task 6.1.

## Sources

Mesh LLM files are linked at commit `213792f5abb6dd0b19e313a282c4636b9a3bce6c`.

- [ARCHITECTURE.md][sk-arch], [SKIPPY_SPLITS.md][sk-splits],
  [FAMILY_CERTIFY.md][sk-certify], [DATA_FLOW.md][sk-flow],
  [LAYER_PACKAGE_REPOS.md][sk-pkg], [GLM_DSA_RUNTIME_CONTRACT.md][sk-dsa]
- [`split-certified.json`][sk-roster], [`family-certified.json`][sk-famcert],
  [llama-parity-candidates.json][sk-cands], [SYNTHETIC_GRAPH_CONTRACTS.md][sk-synth]
- [The patch queue](https://github.com/Mesh-LLM/mesh-llm/tree/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/llama_cpp/patches)
  and [its pin](https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/llama_cpp/upstream.txt)
- [skippy-protocol README][sk-proto], [skippy-correctness README](https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/crates/skippy-correctness/README.md),
  [`single_step.rs`][sk-single], [`binary_messaging.rs`][sk-tcp]
- [Mesh DESIGN.md][mesh-design], [mesh-llm-analytics README][mesh-analytics]
- Other candidates: the repositories linked in [Other candidates](#other-candidates);
  for DwarfStar, its README and
  [docs/DISTRIBUTED.md](https://github.com/antirez/ds4/blob/main/docs/DISTRIBUTED.md)
- Issues [#2663](https://github.com/Mesh-LLM/mesh-llm/issues/2663) and
  [#2726](https://github.com/Mesh-LLM/mesh-llm/issues/2726);
  [releases](https://github.com/Mesh-LLM/mesh-llm/releases)
- dfin: `vendor/patches/0001-layer-range.patch`, `vendor/README.md`,
  `crates/dfin-llama/tests/`, [STATUS.md](../STATUS.md), [protocol.md](protocol.md)

[sk-arch]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/docs/ARCHITECTURE.md
[sk-splits]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/docs/SKIPPY_SPLITS.md
[sk-certify]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/docs/FAMILY_CERTIFY.md
[sk-flow]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/docs/DATA_FLOW.md
[sk-pkg]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/docs/LAYER_PACKAGE_REPOS.md
[sk-dsa]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/docs/specs/GLM_DSA_RUNTIME_CONTRACT.md
[sk-roster]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/crates/skippy-api/src/split-certified.json
[sk-famcert]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/ci/llama-canary/family-certified.json
[sk-cands]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/docs/llama-parity-candidates.json
[sk-synth]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/ci/llama-canary/SYNTHETIC_GRAPH_CONTRACTS.md
[sk-glm5]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/llama_cpp/patches/model_support/0011-models-expose-glm5-next-stage-graph-semantics.patch
[sk-proto]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/crates/skippy-protocol/README.md
[sk-single]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/crates/skippy-correctness/src/runner/single_step.rs
[sk-tcp]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/skippy/crates/skippy-serving/src/binary_transport/binary_messaging.rs
[mesh-design]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/mesh/docs/design/DESIGN.md
[mesh-analytics]: https://github.com/Mesh-LLM/mesh-llm/blob/213792f5abb6dd0b19e313a282c4636b9a3bce6c/mesh/crates/mesh-llm-analytics/README.md
