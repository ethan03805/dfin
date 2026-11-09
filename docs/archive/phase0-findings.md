# Phase 0 findings

Things learned while building the stage runtime that change or sharpen the design.
Dated entries; newest last.

## 2026-11-05: what a stage boundary must carry, per architecture

The design assumed the state passed between stages is the residual stream: one
`n_embd` vector per token (section 3.4 sizes bandwidth from that). Reading the
llama.cpp graph builders of the target models shows this holds for one of the three
target architectures only.

| Architecture (model) | Cross-layer state besides the residual stream | Stage boundary payload per token | Patch status |
|---|---|---|---|
| `llama`, `deepseek2`, `glm4-moe`, `qwen3*` | none | `n_embd` | Supported, verified exact on `llama` |
| `glm-dsa` (GLM-5.3 753B, assumed mapping) | none, but the DSA indexer's top-k is computed at "full indexer" layers and reused by the following "shared" layers | `n_embd` | Supported; a stage must **start at a full-indexer layer**, enforced with a clear error. Routing must respect `is_indexer_full(il)` |
| `glm5-next` (GLM-5.3-Flash 320B-A18B) | manifold hyper-connections: the residual stream is `hc = 4` parallel streams, mixed per layer with Sinkhorn-normalised weights | `4 * n_embd` | **Not yet supported** (loader refuses a range). Extension: define the stage state as the `[n_embd, hc]` block per token; the wire format already carries any `[n_tokens, width]` matrix |
| `kimi-k3` (Kimi K3) | cross-layer attention residuals: every `attn_res_block_size` layers the raw layer input is pushed to a bank; each layer mixes the current prefix sum with every banked vector so far (`res_mix`), and the output head mixes the final bank | `(1 + floor(il / block_size)) * n_embd`, growing with depth; the tail needs the whole bank | **Supported with attention residuals off** (verified exact on a tiny synthetic K3, see below); the loader refuses a range when they are on, and the real checkpoint uses them. Extension: ship prefix sum plus bank; the deepest boundary of a 94-layer model with block size 8 carries about 12 banked vectors, so roughly 13x the bytes of a plain boundary |

Consequences:

- For GLM-5.3 (the launch model) the physics in design section 3 stands as written,
  with one placement rule (indexer-full boundaries).
- For Kimi K3 the inter-stage traffic per token is no longer a constant 6 KB in FP8;
  at the last boundary it is on the order of 80 to 90 KB for `n_embd` 7168 in FP8,
  about 7 ms on a 100 Mbps uplink per token, on top of the RTT. Fewer, fatter stages
  (anchor pods) matter even more for K3 than the design already argued, and the
  replay log for failover (section 7) must store the bank too.
- The patch's state-width generalisation (stage state = `k * n_embd` per token with
  `k` known from the layer index) landed on 6 November 2026 (`docs/phase1-findings.md`
  section 8); both architectures now split. It was originally planned to be done on
  tiny synthetic `glm5-next` and `kimi-k3` models first.

## 2026-11-05: the layer-range patch is small and exact

22 files, 565 insertions in the vendored llama.cpp (`vendor/patches/0001-layer-range.patch`).
The loader skips out-of-range blocks centrally in `create_tensor`; caches skip layers
in their constructors; graph builders iterate `[il_begin, il_end)` and return the
residual stream through `build_stage_out`. On a tiny Llama, three pieces reproduce the
monolithic logits with a maximum absolute difference of 0.0 for prefill, incremental
decode, `seq_rm`, `seq_cp` and `seq_add`, on both CPU and Vulkan; a control that skips
the middle piece changes logits by 2 to 4, so the pieces really compute. Piece sizes sum
exactly to the full model.

The same result holds through the Rust wrapper (`crates/dfin-llama/tests/tiny_model.rs`).

## 2026-11-05: the target architectures run as stages, on tiny synthetic models

No real GLM-5.3 or Kimi K3 weights fit this machine, so `tools/make_tiny_glm_dsa.py` and
`tools/make_tiny_kimi_k3.py` generate structurally faithful random-weight GGUFs (vocab
768, `n_embd` 64, 8 layers) with every per-layer pattern the real models have: GLM-DSA
with three dense layers then MoE (8 experts, 2 used, 1 shared), MLA with absorbed
`k_b`/`v_b`, and a GLM-5.2-style indexer pattern (`1,1,1,0,0,0,1,0` full/shared); Kimi K3
with the KDA/MLA interleave `KKKMKKKM`, one dense layer then latent MoE, SiTU activation,
and nope-only MLA. Both load and run under stock `llama-simple` on CPU and Vulkan with
identical tokens.

`llama-layer-range` (three pieces versus the monolithic model) results:

| Model | Splits | Prompt and incremental decode | `seq_rm` / `seq_cp` / `seq_add` | Notes |
|---|---|---|---|---|
| tiny `glm-dsa` | `2 6`, `1 6`, `1 2` (CPU and Vulkan) | exact, max abs diff 0.0 | exact | `2 5` is refused as designed: layer 5 is a shared-indexer layer, a stage must start at a full-indexer layer. With all layers full, `2 5` passes |
| tiny `kimi-k3` (attention residuals off) | `2 5`, `1 4`, `3 7` (CPU) | exact, max abs diff 0.0 across KDA/MLA and dense/MoE boundaries | `seq_rm` fails on every piece **and would on the monolithic model**: llama.cpp's recurrent memory cannot drop a position range without state snapshots | see below |
| tiny `llama` | `2 5`, `1 7`, `3 4` | exact | exact | |

Two bugs in the patch surfaced only on these models and are fixed:

1. A skipped (out-of-range) tensor requested twice with different shapes, the
   optional-then-required fallback pattern Kimi K3 uses for its conv1d weights, was
   counted twice and failed the tensor-count check. The loader now counts each skipped
   name once.
2. A stage whose range holds only KDA layers (or only MLA layers) still builds both the
   attention and the recurrent graph inputs; the unused ones are never allocated and
   filling them asserted. The hybrid-memory `set_input` paths now skip unallocated inputs.

Consequences for the client:

- **Prefix-cache rollback does not exist for recurrent layers.** The client's prefix cache
  rolls a session back to the common prefix with `seq_rm`; for a KDA model that fails,
  on stages and on the local head and tail alike. Options for Phase 1: enable
  `n_rs_seq` recurrent-state snapshots at turn boundaries (memory cost: one KDA state per
  snapshot per layer), or treat a non-exact prefix match as a miss and re-prefill a fresh
  session. Append-only reuse (the common case for agent turns: the new prompt extends the
  old one) needs no rollback and keeps working.
- **GLM indexer placement rule**: stage boundaries for `glm-dsa` must fall on
  full-indexer layers; the router must read `attention.indexer.types` from the GGUF.
- A synthetic indexer head dimension below 64 breaks context creation (the KV cache builds a
  Hadamard rotation of at least 64), which is a test-model constraint, not a design one.

## 2026-11-05: end-to-end on one machine

`cargo test --workspace`: 28 tests pass (protocol, wrapper split test, stage protocol test
against a real stage process, two end-to-end tests through the client API with two stage
processes and with head-to-tail only). The release benchmark (`docs/logs/phase0-verify-2026-11-05.log`)
generates identical text through the pod and the monolithic model for all three tiny
architectures. Per-hop protocol overhead on localhost is 0.07 to 0.25 ms at `n_embd` 64 in
f16; at GLM-5.3's width (`n_embd` 6144, 12 KB per token per hop in f16) the serialisation
cost will grow but stay far below any WAN round trip. Prefill of the whole prompt through
the pipeline is one `Forward` per stage per chunk, as designed.

Not measured here, and not measurable on this machine: real model speeds, GPU-resident
stages at scale (the Vulkan path works but a 64-wide model is too small to benefit), and
network behaviour beyond loopback.

## 2026-11-05: first real model, GLM-4.7-Flash (30B-A3B) on one machine

Model: `bartowski/zai-org_GLM-4.7-Flash-GGUF`, Q4_K_M, 18.5 GB, stored under the `deepseek2`
architecture (MLA attention, 47 layers, width 2048, one dense layer then 46 MoE layers with
64 experts, 4 routed plus 1 shared). Machine: Ryzen 5 7600X3D, 32 GB DDR5, RX 7800 XT 16 GB
via Vulkan. Placement as the design prescribes for a standard node: attention, router and
shared expert on the GPU, routed experts in RAM (`--cpu-tensors exps`). Split as the design
prescribes for the client: head = layer 0 (the dense layer), tail = layer 46 plus the LM head,
two stages of 23 and 22 layers. Log: `docs/logs/real-model-glm47flash-2026-11-05.log`.

| Mode | Prompt 256, generate 64 | Decode tok/s | Output |
|---|---|---|---|
| Stock llama.cpp (`llama-bench -ngl 99 -ot exps=CPU`) | pp 106 tok/s | **19.9** | reference |
| `dfin` monolithic (same libllama through the Rust wrapper) | prefill 1.42 s | 19.9 | reference |
| `dfin` in-process split, 4 pieces, f16 wire | prefill 1.43 s | 19.5 | **identical tokens** to monolithic |
| `dfin` in-process split, 4 pieces, f32 wire | prefill 1.43 s | 19.5 | identical tokens |
| `dfin` local pod, 4 processes over TCP, f16 | 8.5 s for the whole request | 17.0 | identical text |
| `dfin` local pod, 4 processes over TCP, f32 | 8.4 s | 16.9 | identical text |
| CPU only, monolithic (prompt 128, generate 32) | 1.42 s | 14.8 | reference |
| CPU only, in-process split | 1.45 s | 14.9 | identical tokens |
| CPU only, local pod over TCP | 7.6 s | 12.5 | identical text |

What the numbers say:

- **Splitting costs nothing in-process.** Four pieces in one process decode at the same speed as
  the monolithic model, and the per-piece breakdown is what the physics predicts: the two
  expert-heavy stages take 24.8 and 23.8 ms per token (memory bandwidth for about 1.6 GB of
  active expert weights read from RAM), the dense head 0.2 ms, the tail 1.5 ms, f16
  conversion of the 2048-wide activations 0.6 ms.
- **Exactness holds on a real model.** Greedy tokens match the monolithic run for 64 tokens
  with f16 activations on the wire, not only f32. On this model the f16 rounding of the
  residual stream never flipped an argmax.
- **Four processes on one 6-core CPU lose about 8 ms per token** (59 versus 51 ms). The
  tiny-model runs showed the protocol itself costs 0.1 to 0.25 ms per hop, so this is
  co-location: every piece keeps its own thread pool and Vulkan queue on the same six cores,
  and expert matmuls are memory-bound so idle-but-spinning threads hurt. On a real pod each
  stage has its own machine. The contention experiments below quantify the effect.
- **Partial loading is instant with a warm page cache.** Each stage mapped only its own
  layers and was ready in under 2 s; the head and tail loaded in about 1.3 s each.
- **The chat path works on a real template.** GLM-4.7-Flash's template was recognised by
  llama.cpp's built-in engine; a two-turn conversation through `/v1/chat/completions`
  produced coherent answers and the second turn hit the prefix cache (22 tokens reused).

**Co-location experiments** (`docs/logs/real-model-glm47flash-contention-2026-11-05.log`): with
`--threads 3` per piece instead of the default 6, the four-process pod decodes at 53.6 ms per
token against 50.8 in-process, so the per-hop cost drops from about 4 ms to about 1.4 ms;
passive OpenMP waiting alone changed nothing. The extra cost is therefore the pieces fighting
over six cores and one memory bus, not the wire. A `dfin pod` on one machine should default to
`threads = cores / pieces`; a real pod has one machine per piece and no such contention.

**Memory per piece** (GPU plus experts in RAM, after one request): stage 1:24 working set
4.5 GB, stage 24:46 3.9 GB, client (head, tail, API) 0.8 GB, launcher 70 MB. Private memory is
about 1.3 GB per stage (Vulkan host buffers, KV cache, thread stacks); the expert weights stay
file-backed and shared through the page cache, which is what lets several pieces of an 18 GB
model coexist on a 32 GB machine (8 GB still free with the whole file cached). On a one-piece
node the experts would be read into private memory instead (`--load-mode none` in llama.cpp
terms), as the loader's own warning suggests for speed.

Caveats: a single machine says nothing about network RTT, which the design identifies as the
dominant WAN cost; the stock-llama.cpp baseline and the dfin monolithic number agree, so the
wrapper adds no overhead; llama.cpp warns that CPU tensor overrides with mmap are slower than
reading the file (`--load-mode none`), which was not tested and would be the default on nodes
that run a single piece.

## 2026-11-06: first two-machine pod, over WiFi

Route: this PC runs the head (dense layer 0), a MacBook Air (Apple M4, 16 GB, Metal) serves
layers 1:21, the PC serves layers 21:46 (attention on the RX 7800 XT, experts in RAM), and the
PC runs the tail (layer 46 plus the LM head). The Mac was set up by its own Claude session
from the GitHub repo; the only macOS fixes needed were executable bits on the scripts and an
rpath entry in `.cargo/config.toml` (cargo does not forward a build script's link arguments to
dependent crates). The Mac's stage was ready in seconds and the PC found it by scanning the
LAN for port 7001. Link: WiFi, ping best 3.2 ms, average 20 to 54 ms with spikes above
100 ms. Log: `docs/logs/lan-pod-glm47flash-2026-11-06.log`.

| Mode (GLM-4.7-Flash Q4_K_M, 256-token prompt, 64 generated) | ms per token | tok/s | Output |
|---|---|---|---|
| PC alone, monolithic | 50.5 | 19.8 | reference |
| PC alone, four pieces in one process | 48.6 | 20.6 | identical tokens |
| **PC head and tail, Mac layers 1:21, PC layers 21:46, over WiFi** | 61.6 | 16.2 | identical text |

Per-token breakdown of the two-machine run from the client's decode-only counters
(329 decode steps across four requests):

| Piece | Where | Compute per token | Round trip seen by the client | Network and protocol |
|---|---|---|---|---|
| head 0:1 | PC | 0.7 ms | | |
| stage 1:21 | Mac M4, Metal | **3.7 ms** | 27.0 ms | **23.3 ms** (WiFi; ping floor 3.2 ms) |
| stage 21:46 | PC, Vulkan + CPU experts | 28.3 ms | 28.7 ms | 0.4 ms (loopback) |
| tail 46:47 | PC | 2.3 ms | | |

What this shows:

- **The pipeline is network-bound exactly where the design says it is.** The Mac finishes
  its 20 layers in 3.7 ms and then waits; the client sees 27 ms because every token crosses
  the WiFi link twice. The protocol itself costs 0.4 ms on loopback. On a wired LAN (ping
  under 1 ms) the same route would decode at about 19 tok/s, within 5 percent of the single
  machine.
- **Unified-memory nodes are fast stages.** The M4 runs 20 MoE layers per token about 5 times
  faster per layer than this PC's CPU-expert path (3.7 ms versus 28 ms for 25 layers), because
  its experts sit in 120 GB/s unified memory next to the GPU. This is the anchor argument of
  the design in miniature: the pod would be fastest if the Mac held more layers and the PC
  fewer, bounded by the Mac's 16 GB.
- **Prefix caching works across machines.** The second chat turn reused 28 cached tokens on
  both stages (`prefix_cache.hits` 2) without any rollback failure.
- **Exactness holds over the network.** The 64-token greedy continuation matched the
  monolithic model, f16 activations on the wire, two different GPU backends in one route
  (Metal on the Mac, Vulkan on the PC).

Caveats: the chat template path still leaks GLM's reasoning preamble into the answer, because
llama.cpp's built-in template detection does not know this model's thinking switch; a real
Jinja engine with `enable_thinking` is a client TODO. The Mac ran a stage binary whose
reported compute time did not include the Metal synchronisation that happens when the result
is read, so its 3.7 ms is a lower bound; a bandwidth estimate (about 1.4 GB of expert weights
per token at roughly 120 GB/s) puts the true figure near 12 ms, which would leave about 15 ms
of the 27 ms round trip to the WiFi link. The stage now times the read as well.

## 2026-11-06: cold start is a load-time problem, not a request-time problem

The first request through a freshly started stage was slow: 4.2 s for a 281-token prefill
over the LAN pipeline, and 11 to 27 s for the same prompt through a single local stage
holding 45 layers, against 1.5 s for the monolithic model. Experiments on the PC stage
(`docs/logs/lan-pod-glm47flash-2026-11-06.log` and the session notes):

| Stage setting (45 MoE layers, experts in RAM, attention on GPU) | Load | First prefill (283 tok) | Later prefills | Decode per token | Notes |
|---|---|---|---|---|---|
| mmap (default) | 2 s | 11 to 27 s | 2.4 to 3.4 s | 50 to 65 ms | first request faults 16 GB of expert pages in from the page cache or disk |
| mmap, 3 threads | 2 s | 13.5 s | 2.6 s | 68 to 90 ms | fewer threads cost 30 to 60 percent on decode |
| no mmap (read into private memory) | 10.6 s | 0.7 s | 0.64 s | **0.8 to 1.9 s** | 17.7 GB private on a 32 GB machine: the OS paged 3.4 GB of it out, and each token touches random experts |
| **mmap + mlock** | **43 s** | **1.7 s** | **1.5 s** | **50 to 52 ms** | every page resident at load; working set 17.4 GB; same decode as monolithic |

Consequences for the design:

- **A stage must make its weights resident at load, not at the first request.** `--mlock`
  (llama.cpp's `MMAP_MLOCK`) does this; the design's cold-start gate ("20k-token prefill
  under 30 s") is only meaningful with it. Nodes should report "ready" after locking, and
  the directory should list a node's load time separately from its prefill speed.
- **Reading weights into private memory is only safe with headroom.** On a node that holds
  one piece and little else it is llama.cpp's recommended mode; on a shared machine the
  page cache plus private copies exceed RAM and decode collapses by 20 to 40 times.
- **Thread count is a per-node tuning knob.** Six threads for one piece on a 6-core CPU;
  cores divided by pieces when several pieces share a machine (the local-pod case).
- Warm prefill through a 45-layer stage is still 1.5 s for 283 tokens against 1.4 s
  monolithic, so prefill throughput itself survives the split; the chunk transfer of 1.2 MB
  each way is negligible on a LAN.
- **Two-machine prefill, measured properly** (local stage locked, Mac stage warm): 1.59 s for
  281 tokens end to end, of which 0.62 s is the Mac's round trip (compute plus 1.2 MB each
  way over WiFi) and 0.90 s the PC stage's 25 layers; decode then ran at 59 ms per token
  (16.8 tok/s). The same request against a freshly started, unlocked local stage took 3.4 to
  4.2 s. For the design's cold-start budget the lesson is the same as above: lock at load.

