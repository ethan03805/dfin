#!/usr/bin/env python
"""Generate a tiny random-weight Kimi-K3 (gguf arch "kimi-k3") GGUF for layer-range tests.

Everything is synthetic: weights are Gaussian noise (norm weights = 1), the tokenizer is a toy
SentencePiece-style vocab with byte fallback (copied from make_tiny_llama.py), and the
hyperparameters are shrunk versions of Kimi-K3's. What is kept is the *structure* the
libllama loader and graph exercise (see src/models/kimi-k3.cpp, conversion/kimi_k3.py):
  - hybrid stack: KDA (linear attention, recurrent state) layers interleaved with MLA (full
    attention) layers, default 3 KDA then 1 MLA (MLA at layers 3, 7, ... like
    full_attn_layers = 4, 8, ...). Per-layer head_count_kv encodes it: 0 = KDA, 1 = MLA.
  - KDA: separate q/k/v projections + short conv1d each, full-rank f_a/f_b decay, beta,
    A (= -exp(A_log)), dt bias, full-rank output gate ssm_g, gate_lower_bound (sigmoid decay)
  - MLA: nope-only (no RoPE applied), absorbed k_b/v_b, sigmoid output gate attn_gate;
    plain attn_q by default (--q-lora-rank > 0 switches to q_a/q_b)
  - n_dense leading dense blocks, then latent MoE (router at n_embd, routed experts at
    expert_latent_length via ffn_routed_down/up/norm, exp_probs_b, sigmoid gating,
    shared expert at n_embd), situ activation everywhere
  - cross-layer attention residuals: OFF by default (attn_res.block_size = 0) because the
    loader rejects a layer range when they are on; --attn-res-block-size N turns them on
    (adds attn_res_score / ffn_res_score / output_res_score) for monolithic-only testing.
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
ap.add_argument("--full-attn-every", type=int, default=4,
                help="layer il is MLA iff (il+1) %% N == 0 (K3: 3 KDA then 1 MLA)")
ap.add_argument("--n-ff", type=int, default=128, help="dense FFN width (leading dense blocks)")
ap.add_argument("--n-ff-exp", type=int, default=32, help="per-expert FFN width")
ap.add_argument("--n-expert", type=int, default=8)
ap.add_argument("--n-expert-used", type=int, default=2)
ap.add_argument("--n-expert-shared", type=int, default=1)
ap.add_argument("--n-expert-latent", type=int, default=32, help="0 = experts run at n_embd")
ap.add_argument("--n-dense", type=int, default=1, help="leading dense blocks")
ap.add_argument("--kda-head-dim", type=int, default=16)
ap.add_argument("--conv-kernel", type=int, default=4)
ap.add_argument("--q-lora-rank", type=int, default=0, help="0 = plain attn_q")
ap.add_argument("--kv-lora-rank", type=int, default=32)
ap.add_argument("--qk-nope", type=int, default=16)
ap.add_argument("--qk-rope", type=int, default=8)
ap.add_argument("--v-head", type=int, default=16)
ap.add_argument("--attn-res-block-size", type=int, default=0)
ap.add_argument("--n-ctx", type=int, default=4096)
ap.add_argument("--vocab", type=int, default=768)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--std", type=float, default=0.05)
a = ap.parse_args()

is_mla = [(il + 1) % a.full_attn_every == 0 for il in range(a.n_layer)]

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

# --- metadata (mirrors conversion/kimi_k3.py KimiK3Model.set_gguf_parameters) ----------------
w = gguf.GGUFWriter(a.out, "kimi-k3")
w.add_name("tiny-kimi-k3-test")
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
if a.q_lora_rank > 0:
    w.add_q_lora_rank(a.q_lora_rank)
w.add_kv_lora_rank(a.kv_lora_rank)
w.add_rope_dimension_count(a.qk_rope)                  # nope-only: the "rope" part is not rotated
w.add_key_length(a.kv_lora_rank + a.qk_rope)
w.add_value_length(a.kv_lora_rank)
w.add_key_length_mla(a.qk_nope + a.qk_rope)
w.add_value_length_mla(a.v_head)
w.add_expert_feed_forward_length(a.n_ff_exp)
w.add_expert_shared_count(a.n_expert_shared)
w.add_leading_dense_block_count(a.n_dense)
w.add_expert_weights_scale(2.5)
w.add_expert_weights_norm(True)
w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
if a.n_expert_latent > 0:
    w.add_expert_latent_length(a.n_expert_latent)
w.add_activation_situ_beta(4.0)
w.add_activation_situ_linear_beta(8.0)
w.add_attn_res_block_size(a.attn_res_block_size)
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
dkda = a.kda_head_dim
d_inner = H * dkda
dk = a.qk_nope + a.qk_rope
L = a.n_expert_latent if a.n_expert_latent > 0 else E
res = a.attn_res_block_size > 0

w.add_tensor("token_embd.weight", rnd((a.vocab, E), 1.0))
for il in range(a.n_layer):
    p = f"blk.{il}."
    w.add_tensor(p + "attn_norm.weight", ones(E))
    w.add_tensor(p + "ffn_norm.weight", ones(E))
    if res:
        w.add_tensor(p + "attn_res_score.weight", rnd((E,), 0.1))
        w.add_tensor(p + "ffn_res_score.weight", rnd((E,), 0.1))
    if not is_mla[il]:
        # KDA layer
        for x in "qkv":
            w.add_tensor(p + f"ssm_conv1d_{x}.weight", rnd((1, d_inner, 1, a.conv_kernel), 0.3))  # ne {d_conv,1,d_inner,1}
            w.add_tensor(p + f"attn_{x}.weight", rnd((d_inner, E)))
        w.add_tensor(p + "ssm_f_a.weight", rnd((dkda, E)))
        w.add_tensor(p + "ssm_f_b.weight", rnd((d_inner, dkda)))
        w.add_tensor(p + "ssm_beta.weight", rnd((H, E)))
        w.add_tensor(p + "ssm_a", -np.exp(rnd((H,), 0.5)))        # stores -exp(A_log)
        w.add_tensor(p + "ssm_dt.bias", rnd((d_inner,), 0.1))
        w.add_tensor(p + "ssm_g.weight", rnd((d_inner, E)))
        w.add_tensor(p + "ssm_norm.weight", ones(dkda))
        w.add_tensor(p + "attn_output.weight", rnd((E, d_inner)))
    else:
        # MLA layer (nope-only, output gate)
        if a.q_lora_rank > 0:
            w.add_tensor(p + "attn_q_a_norm.weight", ones(a.q_lora_rank))
            w.add_tensor(p + "attn_q_a.weight", rnd((a.q_lora_rank, E)))
            w.add_tensor(p + "attn_q_b.weight", rnd((H * dk, a.q_lora_rank), 0.2))
        else:
            w.add_tensor(p + "attn_q.weight", rnd((H * dk, E)))
        w.add_tensor(p + "attn_kv_a_norm.weight", ones(a.kv_lora_rank))
        w.add_tensor(p + "attn_kv_a_mqa.weight", rnd((a.kv_lora_rank + a.qk_rope, E)))
        w.add_tensor(p + "attn_k_b.weight", rnd((H, a.kv_lora_rank, a.qk_nope), 0.2))   # ne {nope, kv_lora, H}
        w.add_tensor(p + "attn_v_b.weight", rnd((H, a.v_head, a.kv_lora_rank), 0.2))    # ne {kv_lora, v, H}
        w.add_tensor(p + "attn_gate.weight", rnd((H * a.v_head, E)))
        w.add_tensor(p + "attn_output.weight", rnd((E, H * a.v_head)))
    if il < a.n_dense:
        w.add_tensor(p + "ffn_gate.weight", rnd((a.n_ff, E)))
        w.add_tensor(p + "ffn_up.weight", rnd((a.n_ff, E)))
        w.add_tensor(p + "ffn_down.weight", rnd((E, a.n_ff)))
    else:
        X, F = a.n_expert, a.n_ff_exp
        Fs = F * max(a.n_expert_shared, 1)
        w.add_tensor(p + "ffn_gate_inp.weight", rnd((X, E), 0.5))
        w.add_tensor(p + "exp_probs_b.bias", rnd((X,), 0.1))
        w.add_tensor(p + "ffn_gate_exps.weight", rnd((X, F, L)))
        w.add_tensor(p + "ffn_up_exps.weight", rnd((X, F, L)))
        w.add_tensor(p + "ffn_down_exps.weight", rnd((X, L, F)))
        if a.n_expert_latent > 0:
            w.add_tensor(p + "ffn_routed_down.weight", rnd((L, E)))
            w.add_tensor(p + "ffn_routed_up.weight", rnd((E, L)))
            w.add_tensor(p + "ffn_routed_norm.weight", ones(L))
        if a.n_expert_shared > 0:
            w.add_tensor(p + "ffn_gate_shexp.weight", rnd((Fs, E)))
            w.add_tensor(p + "ffn_up_shexp.weight", rnd((Fs, E)))
            w.add_tensor(p + "ffn_down_shexp.weight", rnd((E, Fs)))
w.add_tensor("output_norm.weight", ones(E))
w.add_tensor("output.weight", rnd((a.vocab, E), 0.5))
if res:
    w.add_tensor("output_res_score.weight", rnd((E,), 0.1))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {a.out}: kimi-k3 n_embd={E} n_layer={a.n_layer} n_head={H} dense={a.n_dense} "
      f"experts={a.n_expert}/{a.n_expert_used}+{a.n_expert_shared} latent={a.n_expert_latent} "
      f"layers={''.join('M' if m else 'K' for m in is_mla)} attn_res={a.attn_res_block_size} vocab={a.vocab}")
