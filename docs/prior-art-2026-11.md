# Prior art check, 5 November 2026

What exists today for running one model's layers across several machines on ggml, and
what that meant for the Phase 0 stage runtime. Compiled from the project pages and
source trees on the date above; liveness is "last commit" as seen then.

## Decision this informed

Mainline llama.cpp has no way to run only layers `[a, b)` of a model. The irreducible
surface for adding it is small and has been built at least three times independently:

1. a loader filter by block id, with the embedding on the first piece and the LM head
   on the last;
2. a graph entry point that accepts a hidden state (the existing `llama_batch.embd`
   path) and an early exit that returns the residual stream before the final norm;
3. per-piece `llama_context` and KV cache, which fall out naturally;
4. an external transport.

We implemented exactly that as a vendored patch (`vendor/README.md`) rather than
re-implementing architectures on raw ggml. Every architecture llama.cpp supports then
works as a stage, including `glm-dsa`, `glm5-next` and `kimi-k3`.

## Projects

| Project | Mechanism | Transport | Models | Liveness | License | Verdict |
|---|---|---|---|---|---|---|
| **Unsloth llama.cpp PR #180** | Loader skips blocks outside `LLAMA_PP_IL_BEG/END`; second stage takes hidden state via `batch.embd`; small TCP stage tool; bit-exact on Qwen3 | TCP | qwen35, qwen3moe wired | Closed unmerged 2026-09-04, "superseded by #18626" | MIT | The minimal design we followed |
| **Mesh LLM / skippy** | Patch queue (16 core + ~83 generated per-family patches) on llama.cpp pinned 2026-10-15; contiguous layer ranges per peer; layer-sliced GGUF "packages" on Hugging Face; final stage returns tokens straight to stage 0 | QUIC via Iroh (encrypted, relays); discovery via Nostr | Certified roster: Qwen, Llama, Gemma, DeepSeek, GLM, MiniMax, many more | Commit 2026-11-05 | Apache-2.0 | Closest living relative. Their docs list concurrent admission and suffix-cache reuse as open gaps. Worth tracking; possibly worth converging with later |
| **prima.cpp** (arXiv 2504.08791, ICLR 2026) | Ring pipeline with per-device layer windows (`-lw`), round-robin wrap-around, Halda ILP scheduler, mmap prefetch | ZeroMQ PUSH/PULL, raw f32 | Llama family and Qwen2 only | Fork of llama.cpp from 2024-10-07, last code 2025-07-17; GitHub repo gone, lives on Gitee | MIT | Dead end to reuse; good ideas (prefetch, scheduler) |
| **llama.cpp RPC backend** | One host holds the GGUF and the graph; `rpc-server`s are dumb ggml backends; server-to-server copies fall back to host copies | TCP, optional RDMA (RoCEv2, Thunderbolt 5) | all | Mainline; README: proof of concept, "never run on an open network" | MIT | Not a decentralised pipeline: every boundary crosses the host, host needs full model |
| **distributed-llama** | Tensor parallel (root plus 2^n−1 workers, max nodes = KV heads), own `.m` format | TCP | Llama 3.x, Qwen3 | Commit 2026-11-01 | MIT | Different split; needs LAN-class bandwidth |
| **exo** | Pipeline and tensor sharding on MLX only | RDMA/Thunderbolt, Wi-Fi/Ethernet | MLX models | Commit 2026-11-05 | Apache-2.0 | Apple-only engine |
| **Petals** | Transformer blocks over a DHT, PyTorch | libp2p/hivemind | BLOOM/Llama era | Last commit 2024-08-25 | MIT | Dead; design ideas live on |
| **BloomBee** | Petals-style, PyTorch | libp2p | HF models | Commit 2026-10-21 | Apache-2.0 | Not ggml; lossless activation compression idea adopted in the design |
| **Kalavai** | Orchestration over vLLM, llama.cpp, Ray | n/a | n/a | Commit 2026-11-03 | Apache-2.0 | Orchestration layer, not a split engine |

Not found or unverifiable: Wavefy, SharedLLM (blog only), DistLLM.

## Model support in mainline llama.cpp (as of commit 45b6604b, 5 Nov 2026)

| Model | llama.cpp arch | Status | Notes |
|---|---|---|---|
| GLM-5.3-Flash | `glm5-next` | Merged 2026-09-30 (PR #27773) | 34 KDA linear layers + 11 DSA layers. MTP merged 2026-10-07 (`--spec-type draft-mtp`). Multiple sequences in one unified KV cache reportedly fail (pooled indexer); use `--no-kv-unified` or one sequence per stream |
| GLM-5.3 (753B) | `glm-dsa` (assumed; HF `model_type` is `glm_moe_dsa`) | Arch merged Feb 2026, indexer working since Jul 2026 for GLM-5.2 | **Unverified** that 5.3 loads and runs in mainline; one ik_llama quant repo claims standard llama.cpp cannot run its quants. Must be tested on real hardware |
| Kimi K3 | `kimi-k3` | Text merged 2026-08-15 (PR #26185) | Vision only in the Unsloth fork. 69 KDA + 24 gated-MLA layers: per-session state on a stage is a fixed-size recurrent state plus an MLA cache |

This supersedes the design document's assumption that Kimi K3 needs the Unsloth fork.

## Consequences for the design

- Both target models use **KDA (Kimi Delta Attention)** linear layers for most of their
  depth. A stage's per-session state is mostly fixed-size recurrent state, which is
  cheaper than KV and makes replay-log failover (design section 7) simpler, but
  rollback for prefix caching needs recurrent-state snapshots (`n_rs_seq` in
  `llama_context_params`) rather than KV truncation. Phase 0 tests this on tiny
  synthetic models of each architecture before any real weights are involved.
- The GLM pooled-indexer multi-sequence issue means stages serving GLM-5.x must run
  with `kv_unified = false`, which the stage server already does.
- The RPC backend's new async APIs (PR #18626) are the upstream direction for
  multi-host pipelines; worth re-reading before Phase 1 to see whether stage-to-stage
  transfer can be expressed with them instead of our own protocol.
