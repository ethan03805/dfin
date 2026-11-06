#!/usr/bin/env python
"""Generate a tiny random-weight GLM-DSA (GLM-5.x, gguf arch "glm-dsa") GGUF for layer-range tests.

Everything is synthetic: weights are Gaussian noise (norm weights = 1), the tokenizer is a toy
SentencePiece-style vocab with byte fallback (copied from make_tiny_llama.py), and the
hyperparameters are shrunk versions of GLM-5.2's. What is kept from the real model is the
*structure* the libllama loader and graph exercise:
  - MLA attention (q_lora/kv_lora, nope/rope split, absorbed k_b/v_b, MQA with 1 KV head)
  - n_dense leading dense-FFN blocks, then MoE blocks (routed experts + shared expert,
    sigmoid gating, exp_probs_b selection bias)
  - DSA lightning indexer with a GLM-5.2-style "full"/"shared" indexer pattern
    (default for 8 layers: 1,1,1,0,0,0,1,0 = first three full, then every 4th from layer 2).
    Shared layers reuse the previous full layer's top-k and carry no indexer tensors.
  - no NextN/MTP block (nextn_predict_layers = 0)
The model is numerically meaningless but llama.cpp loads and runs it.
Note: a layer range [a, b) must start at a full-indexer layer (loader check).
"""
import argparse
import numpy as np
import gguf

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n-embd", type=int, default=64)
ap.add_argument("--n-layer", type=int, default=8)
ap.add_argument("--n-head", type=int, default=4)
ap.add_argument("--n-ff", type=int, default=128, help="dense FFN width (leading dense blocks)")
ap.add_argument("--n-ff-exp", type=int, default=32, help="per-expert FFN width")
ap.add_argument("--n-expert", type=int, default=8)
ap.add_argument("--n-expert-used", type=int, default=2)
ap.add_argument("--n-expert-shared", type=int, default=1)
ap.add_argument("--n-dense", type=int, default=3, help="leading dense blocks (GLM-5: 3)")
ap.add_argument("--q-lora-rank", type=int, default=32)
ap.add_argument("--kv-lora-rank", type=int, default=32)
ap.add_argument("--qk-nope", type=int, default=16)
ap.add_argument("--qk-rope", type=int, default=8)
ap.add_argument("--v-head", type=int, default=16)
ap.add_argument("--idx-head", type=int, default=4, help="indexer head count")
ap.add_argument("--idx-dim", type=int, default=64, help="indexer head dim (power of 2, >= 64 for the KV-cache Hadamard rotation, >= qk-rope)")
ap.add_argument("--idx-top-k", type=int, default=8)
ap.add_argument("--idx-types", default=None,
                help="comma list of 1(full)/0(shared) per layer; default GLM-5.2-like pattern")
ap.add_argument("--n-ctx", type=int, default=4096)
ap.add_argument("--vocab", type=int, default=768)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--std", type=float, default=0.05)
a = ap.parse_args()

if a.idx_types:
    idx_full = [int(x) for x in a.idx_types.split(",")]
else:
    idx_full = [1 if (il < 3 or (il - 2) % 4 == 0) else 0 for il in range(a.n_layer)]
assert len(idx_full) == a.n_layer and idx_full[0] == 1, "need one type per layer, layer 0 full"
assert a.idx_dim >= a.qk_rope and a.idx_dim >= 64 and (a.idx_dim & (a.idx_dim - 1)) == 0

rng = np.random.default_rng(a.seed)

def rnd(shape, std=a.std):
    return (rng.standard_normal(shape) * std).astype(np.float32)

def ones(n):
    return np.ones(n, np.float32)

# --- tokenizer: SentencePiece-style vocab with byte fallback -------------------------------
T = gguf.TokenType
tokens, types, scores = [], [], []
def add(tok, ty, score):
    tokens.append(tok); types.append(ty); scores.append(float(score))
add("<unk>", T.UNKNOWN, 0.0)
add("<s>", T.CONTROL, 0.0)
add("</s>", T.CONTROL, 0.0)
for i in range(256):
    add(f"<0x{i:02X}>", T.BYTE, 0.0)
letters = "abcdefghijklmnopqrstuvwxyz"
words = ["the", "of", "and", "to", "in", "is", "it", "that", "for", "on", "with", "as", "was",
         "at", "by", "an", "be", "this", "are", "from", "or", "one", "had", "not", "but", "what",
         "all", "were", "we", "when", "your", "can", "said", "there", "use", "each", "which", "she",
         "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them",
         "these", "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look",
         "two", "more", "write", "go", "see", "number", "no", "way", "could", "people", "my", "than",
         "first", "water", "been", "call", "who", "oil", "its", "now", "find", "long", "down", "day",
         "did", "get", "come", "made", "may", "part", "hello", "world", "model", "network", "node"]
score = -1.0
for c in letters:
    add("▁" + c, T.NORMAL, score); score -= 1.0
for c in letters:
    add(c, T.NORMAL, score); score -= 1.0
for c in "0123456789.,!?;:'\"()-":
    add(c, T.NORMAL, score); score -= 1.0
add("▁", T.NORMAL, score); score -= 1.0
for wd in words:
    add("▁" + wd, T.NORMAL, score); score -= 1.0
    add(wd, T.NORMAL, score); score -= 1.0
while len(tokens) < a.vocab:
    add(f"<pad_{len(tokens)}>", T.UNUSED, -1e9)
assert len(tokens) == a.vocab, f"vocab {a.vocab} too small for {len(tokens)} tokens"

# --- metadata (mirrors conversion/deepseek.py DeepseekV2Model + conversion/glm.py GlmMoeDsaModel)
w = gguf.GGUFWriter(a.out, "glm-dsa")
w.add_name("tiny-glm-dsa-test")
w.add_context_length(a.n_ctx)                 # < 1M: no GLM-5.2 default indexer table
w.add_embedding_length(a.n_embd)
w.add_block_count(a.n_layer)
w.add_feed_forward_length(a.n_ff)
w.add_head_count(a.n_head)
w.add_head_count_kv(1)                        # MLA absorbed -> MQA
w.add_rope_freq_base(10000.0)
w.add_layer_norm_rms_eps(1e-5)
w.add_expert_count(a.n_expert)
w.add_expert_used_count(a.n_expert_used)
w.add_leading_dense_block_count(a.n_dense)
w.add_vocab_size(a.vocab)
w.add_q_lora_rank(a.q_lora_rank)
w.add_kv_lora_rank(a.kv_lora_rank)
w.add_key_length(a.kv_lora_rank + a.qk_rope)
w.add_value_length(a.kv_lora_rank)
w.add_key_length_mla(a.qk_nope + a.qk_rope)
w.add_value_length_mla(a.v_head)
w.add_expert_feed_forward_length(a.n_ff_exp)
w.add_expert_shared_count(a.n_expert_shared)
w.add_expert_weights_scale(2.5)
w.add_expert_weights_norm(True)
w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
w.add_rope_dimension_count(a.qk_rope)
w.add_nextn_predict_layers(0)
w.add_indexer_head_count(a.idx_head)
w.add_indexer_key_length(a.idx_dim)
w.add_indexer_top_k(a.idx_top_k)
w.add_indexer_types([bool(x) for x in idx_full])
w.add_file_type(gguf.LlamaFileType.ALL_F32)

w.add_tokenizer_model("llama")
w.add_token_list(tokens)
w.add_token_scores(scores)
w.add_token_types(types)
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_unk_token_id(0)
w.add_add_bos_token(True)
w.add_add_eos_token(False)

# --- weights (numpy shapes are the reverse of the ggml ne[] in load_tensors) ----------------
E, H = a.n_embd, a.n_head
dk = a.qk_nope + a.qk_rope
w.add_tensor("token_embd.weight", rnd((a.vocab, E), 1.0))
for il in range(a.n_layer):
    p = f"blk.{il}."
    w.add_tensor(p + "attn_norm.weight", ones(E))
    w.add_tensor(p + "attn_q_a_norm.weight", ones(a.q_lora_rank))
    w.add_tensor(p + "attn_kv_a_norm.weight", ones(a.kv_lora_rank))
    w.add_tensor(p + "attn_q_a.weight", rnd((a.q_lora_rank, E)))
    w.add_tensor(p + "attn_q_b.weight", rnd((H * dk, a.q_lora_rank), 0.2))
    w.add_tensor(p + "attn_kv_a_mqa.weight", rnd((a.kv_lora_rank + a.qk_rope, E)))
    w.add_tensor(p + "attn_k_b.weight", rnd((H, a.kv_lora_rank, a.qk_nope), 0.2))   # ne {nope, kv_lora, H}
    w.add_tensor(p + "attn_v_b.weight", rnd((H, a.v_head, a.kv_lora_rank), 0.2))    # ne {kv_lora, v, H}
    w.add_tensor(p + "attn_output.weight", rnd((E, H * a.v_head)))
    w.add_tensor(p + "ffn_norm.weight", ones(E))
    if idx_full[il]:
        w.add_tensor(p + "indexer.k_norm.weight", ones(a.idx_dim))
        w.add_tensor(p + "indexer.k_norm.bias", np.zeros(a.idx_dim, np.float32))
        w.add_tensor(p + "indexer.proj.weight", rnd((a.idx_head, E)))
        w.add_tensor(p + "indexer.attn_k.weight", rnd((a.idx_dim, E)))
        w.add_tensor(p + "indexer.attn_q_b.weight", rnd((a.idx_head * a.idx_dim, a.q_lora_rank), 0.2))
    if il < a.n_dense:
        w.add_tensor(p + "ffn_gate.weight", rnd((a.n_ff, E)))
        w.add_tensor(p + "ffn_up.weight", rnd((a.n_ff, E)))
        w.add_tensor(p + "ffn_down.weight", rnd((E, a.n_ff)))
    else:
        X, F = a.n_expert, a.n_ff_exp
        Fs = F * a.n_expert_shared
        w.add_tensor(p + "ffn_gate_inp.weight", rnd((X, E), 0.5))
        w.add_tensor(p + "exp_probs_b.bias", rnd((X,), 0.1))
        w.add_tensor(p + "ffn_gate_exps.weight", rnd((X, F, E)))
        w.add_tensor(p + "ffn_up_exps.weight", rnd((X, F, E)))
        w.add_tensor(p + "ffn_down_exps.weight", rnd((X, E, F)))
        w.add_tensor(p + "ffn_gate_shexp.weight", rnd((Fs, E)))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd((Fs, E)))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd((E, Fs)))
w.add_tensor("output_norm.weight", ones(E))
w.add_tensor("output.weight", rnd((a.vocab, E), 0.5))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {a.out}: glm-dsa n_embd={E} n_layer={a.n_layer} n_head={H} dense={a.n_dense} "
      f"experts={a.n_expert}/{a.n_expert_used}+{a.n_expert_shared} indexer_full={idx_full} vocab={a.vocab}")
