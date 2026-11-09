# Vendored dependencies

## llama.cpp

`vendor/llama.cpp` is a git submodule of mainline [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
pinned at commit `45b6604b20f17b7d7638042306c0cd1b6af5999c` (5 November 2026). The
**layer-range patch** in `vendor/patches/` is not committed to the submodule: after
`git submodule update --init`, `scripts/apply-llama-patch.sh` (or `.ps1`) applies it to
the working tree, which is why `git status` shows the submodule as modified.

### The layer-range patch

Mainline libllama always loads and runs every layer. The patch adds:

- `llama_model_params.layer_begin` / `layer_end` (defaults `0` / `-1` = all layers) and
  read accessors `llama_model_layer_begin()` / `llama_model_layer_end()`.
- Loader: tensors of blocks outside the range are neither created nor read. `token_embd`
  is skipped when `layer_begin > 0`; `output_norm` and `output` are skipped when
  `layer_end < n_layer`.
- Graph: the per-architecture builders iterate only the owned layers. A piece without
  the embedding takes hidden states through `llama_batch.embd`; a piece without the LM
  head returns the raw residual stream at the boundary as its "embeddings"
  (`llama_get_embeddings`, pooling type NONE).
- KV cache: allocated only for owned layers.

- Two robustness fixes the tiny GLM-DSA and Kimi K3 models exposed: a skipped tensor that an
  architecture requests twice (optional-then-required fallback) is counted once, and hybrid
  memory inputs that a stage's layer range never uses are left unfilled instead of asserting.

With default parameters behaviour is unchanged. The patch is exercised by
`examples/layer-range` inside the checkout (`llama-layer-range.exe <model.gguf> <a> <b>` runs
the model monolithically and as three pieces and compares logits) and by
`crates/dfin-llama/tests/tiny_model.rs`. Verified exact on tiny `llama`, `glm-dsa` and
`kimi-k3` models, CPU and Vulkan. Architectures whose state between layers is wider than
one residual vector (`glm5-next` hyper-connection streams, Kimi K3 attention-residual
banks) expose the width through `llama_model_n_embd_inp` and `llama_model_n_embd_out`:
a stage takes and produces rows of that width, and `stage_state_mult(il)` on the model
class says how many `n_embd` vectors cross the boundary before layer `il`.

### Rebuilding

```powershell
.\scripts\build-llama.ps1          # Windows, Vulkan + CPU
```

```bash
scripts/build-llama.sh             # Linux/macOS
```

After any change to `vendor/llama.cpp/include/llama.h`, regenerate the Rust bindings:

```powershell
.\scripts\gen-bindings.ps1
```

### Re-applying the patch to a newer llama.cpp

```bash
git -C vendor/llama.cpp fetch origin && git -C vendor/llama.cpp checkout <new-commit>
git -C vendor/llama.cpp apply ../patches/0001-layer-range.patch
```

Then rebuild, regenerate bindings and run `cargo test -p dfin-llama`.
