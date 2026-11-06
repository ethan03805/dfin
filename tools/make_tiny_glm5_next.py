#!/usr/bin/env python
"""Generate a tiny random-weight GLM5-Next (GLM-5.3-Flash, gguf arch "glm5-next") GGUF for
layer-range tests.

Everything is synthetic: weights are Gaussian noise (norm weights = 1), the tokenizer is a toy
SentencePiece-style vocab with byte fallback, and the hyperparameters are shrunk. What is
kept from the real model is the *structure* the libllama loader and graph exercise:
  - hybrid layers: KDA (linear attention, head_count_kv 0) and nope-only MLA (no rope part,
    rope dimension count 0) every
    --full-attn-every layers, each MLA layer with a k-pool DSA indexer (full on every layer)
  - manifold hyper-connections: hc = 4 residual streams mixed per layer (hc_attn_*, hc_ffn_*),
    which is what makes the state between layers 4 * n_embd wide
  - n_dense leading dense-FFN blocks, then MoE (routed + shared experts, sigmoid gating)
  - no NextN/MTP block
The model is numerically meaningless but llama.cpp loads and runs it.
"""
import argparse
import numpy as np
import gguf

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n-embd", type=int, default=64)
ap.add_argument("--n-layer", type=int, default=8)
ap.add_argument("--n-head", type=int, default=4, help="heads for both KDA and MLA layers")
ap.add_argument("--full-attn-every", type=int, default=4, help="layer il is MLA when (il + 1) %% N == 0")
ap.add_argument("--n-ff", type=int, default=128)
ap.add_argument("--n-ff-exp", type=int, default=32)
ap.add_argument("--n-expert", type=int, default=8)
ap.add_argument("--n-expert-used", type=int, default=2)
ap.add_argument("--n-expert-shared", type=int, default=1)
ap.add_argument("--n-dense", type=int, default=1)
ap.add_argument("--kda-head-dim", type=int, default=16)
ap.add_argument("--conv-kernel", type=int, default=4)
ap.add_argument("--q-lora-rank", type=int, default=32)
ap.add_argument("--kv-lora-rank", type=int, default=32)
ap.add_argument("--qk-nope", type=int, default=16)
ap.add_argument("--v-head", type=int, default=16)
ap.add_argument("--idx-head", type=int, default=4)
ap.add_argument("--idx-dim", type=int, default=64)
ap.add_argument("--idx-top-k", type=int, default=8)
ap.add_argument("--kpool", type=int, default=2)
ap.add_argument("--hc", type=int, default=4)
ap.add_argument("--n-ctx", type=int, default=4096)
ap.add_argument("--vocab", type=int, default=768)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--std", type=float, default=0.05)
a = ap.parse_args()

assert a.idx_top_k % a.kpool == 0 and a.kpool > 1
is_mla = [(il + 1) % a.full_attn_every == 0 for il in range(a.n_layer)]
assert any(is_mla), "need at least one MLA layer"

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
         "did", "get", "come", "made", "may", "part", "model", "network", "node", "hello", "world"]
for c in letters:
    add(c, T.NORMAL, -1.0)
    add("▁" + c, T.NORMAL, -1.0)
for i, wd in enumerate(words):
    add("▁" + wd, T.NORMAL, -2.0 - i * 0.01)
    add(wd, T.NORMAL, -3.0 - i * 0.01)
add("▁", T.NORMAL, -5.0)
while len(tokens) < a.vocab:
    add(f"<pad_{len(tokens)}>", T.UNUSED, -1e9)
assert len(tokens) == a.vocab, f"vocab {a.vocab} too small for {len(tokens)} tokens"

# --- metadata --------------------------------------------------------------------------------
arch = "glm5-next"
w = gguf.GGUFWriter(a.out, arch)
w.add_name("tiny-glm5-next-test")
w.add_context_length(a.n_ctx)
w.add_embedding_length(a.n_embd)
w.add_block_count(a.n_layer)
w.add_feed_forward_length(a.n_ff)
w.add_head_count(a.n_head)
w.add_head_count_kv([1 if m else 0 for m in is_mla])   # 0 marks a KDA (recurrent) layer
w.add_layer_norm_rms_eps(1e-5)
w.add_expert_count(a.n_expert)
w.add_expert_used_count(a.n_expert_used)
w.add_vocab_size(a.vocab)
w.add_ssm_conv_kernel(a.conv_kernel)
w.add_kda_head_dim(a.kda_head_dim)
w.add_kda_gate_lower_bound(-5.0)
w.add_q_lora_rank(a.q_lora_rank)
w.add_kv_lora_rank(a.kv_lora_rank)
w.add_rope_dimension_count(0)                       # nope-only MLA
w.add_key_length(a.kv_lora_rank)
w.add_value_length(a.kv_lora_rank)
w.add_key_length_mla(a.qk_nope)
w.add_value_length_mla(a.v_head)
w.add_expert_feed_forward_length(a.n_ff_exp)
w.add_expert_shared_count(a.n_expert_shared)
w.add_leading_dense_block_count(a.n_dense)
w.add_expert_weights_scale(2.5)
w.add_expert_weights_norm(True)
w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
w.add_nextn_predict_layers(0)
w.add_indexer_head_count(a.idx_head)
w.add_indexer_key_length(a.idx_dim)
w.add_indexer_top_k(a.idx_top_k)
w.add_indexer_kpool(a.kpool)
w.add_indexer_types([True] * a.n_layer)
w.add_uint32(f"{arch}.hyper_connection.count", a.hc)
w.add_uint32(f"{arch}.hyper_connection.sinkhorn_iterations", 3)
w.add_float32(f"{arch}.hyper_connection.epsilon", 1e-6)
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
E, H, hc = a.n_embd, a.n_head, a.hc
dkda = a.kda_head_dim
d_inner = H * dkda
dk = a.qk_nope
hc_mix = (2 + hc) * hc

w.add_tensor("token_embd.weight", rnd((a.vocab, E), 1.0))
for il in range(a.n_layer):
    p = f"blk.{il}."
    w.add_tensor(p + "attn_norm.weight", ones(E))
    w.add_tensor(p + "ffn_norm.weight", ones(E))
    for kind in ("attn", "ffn"):
        w.add_tensor(p + f"hc_{kind}_fn.weight", rnd((hc_mix, hc * E), 0.1))   # ne {hc*E, hc_mix}
        w.add_tensor(p + f"hc_{kind}_base.weight", rnd((hc_mix,), 0.1))
        w.add_tensor(p + f"hc_{kind}_scale.weight", np.array([0.1, 0.1, 0.1], np.float32))
    if not is_mla[il]:
        for x in "qkv":
            w.add_tensor(p + f"ssm_conv1d_{x}.weight", rnd((1, d_inner, 1, a.conv_kernel), 0.3))
            w.add_tensor(p + f"attn_{x}.weight", rnd((d_inner, E)))
        w.add_tensor(p + "ssm_f_a.weight", rnd((dkda, E)))
        w.add_tensor(p + "ssm_f_b.weight", rnd((d_inner, dkda)))
        w.add_tensor(p + "ssm_beta.weight", rnd((H, E)))
        w.add_tensor(p + "ssm_a", -np.exp(rnd((H,), 0.5)))
        w.add_tensor(p + "ssm_dt.bias", rnd((d_inner,), 0.1))
        w.add_tensor(p + "ssm_g_a.weight", rnd((dkda, E)))
        w.add_tensor(p + "ssm_g_b.weight", rnd((d_inner, dkda)))
        w.add_tensor(p + "ssm_norm.weight", ones(dkda))
        w.add_tensor(p + "attn_output.weight", rnd((E, d_inner)))
    else:
        w.add_tensor(p + "attn_q_a_norm.weight", ones(a.q_lora_rank))
        w.add_tensor(p + "attn_kv_a_norm.weight", ones(a.kv_lora_rank))
        w.add_tensor(p + "attn_q_a.weight", rnd((a.q_lora_rank, E)))
        w.add_tensor(p + "attn_q_b.weight", rnd((H * dk, a.q_lora_rank), 0.2))
        w.add_tensor(p + "attn_kv_a_mqa.weight", rnd((a.kv_lora_rank, E)))
        w.add_tensor(p + "attn_k_b.weight", rnd((H, a.kv_lora_rank, a.qk_nope), 0.2))
        w.add_tensor(p + "attn_v_b.weight", rnd((H, a.v_head, a.kv_lora_rank), 0.2))
        w.add_tensor(p + "attn_output.weight", rnd((E, H * a.v_head)))
        w.add_tensor(p + "indexer.k_norm.weight", ones(a.idx_dim))
        w.add_tensor(p + "indexer.k_norm.bias", np.zeros(a.idx_dim, np.float32))
        w.add_tensor(p + "indexer.proj.weight", rnd((a.idx_head, E)))
        w.add_tensor(p + "indexer.attn_k.weight", rnd((a.idx_dim, E)))
        w.add_tensor(p + "indexer.attn_q_b.weight", rnd((a.idx_head * a.idx_dim, a.q_lora_rank), 0.2))
        w.add_tensor(p + "indexer_compressor_gate.weight", rnd((a.idx_dim, E)))
        w.add_tensor(p + "indexer_compressor_ape.weight", rnd((a.kpool, a.idx_dim)))
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
print(f"wrote {a.out}: glm5-next n_embd={E} n_layer={a.n_layer} hc={hc} n_head={H} dense={a.n_dense} "
      f"experts={a.n_expert}/{a.n_expert_used}+{a.n_expert_shared} "
      f"layers={''.join('M' if m else 'K' for m in is_mla)} kpool={a.kpool} vocab={a.vocab}")
