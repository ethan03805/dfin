#!/usr/bin/env python
"""Generate a tiny random-weight Llama-architecture GGUF for pipeline correctness tests.

The model is numerically meaningless but structurally valid: llama.cpp loads and runs it,
so a pipelined run (head + stages + tail) can be compared bit-for-bit-ish against a
monolithic run of the same file. No download required.
"""
import argparse
import numpy as np
import gguf

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n-embd", type=int, default=64)
ap.add_argument("--n-layer", type=int, default=8)
ap.add_argument("--n-head", type=int, default=4)
ap.add_argument("--n-head-kv", type=int, default=2)
ap.add_argument("--n-ff", type=int, default=128)
ap.add_argument("--n-ctx", type=int, default=1024)
ap.add_argument("--vocab", type=int, default=768)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--std", type=float, default=0.05)
a = ap.parse_args()

head_dim = a.n_embd // a.n_head
n_embd_kv = a.n_head_kv * head_dim
rng = np.random.default_rng(a.seed)

def rnd(shape, std=a.std):
    return (rng.standard_normal(shape) * std).astype(np.float32)

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

w = gguf.GGUFWriter(a.out, "llama")
w.add_name("tiny-llama-test")
w.add_context_length(a.n_ctx)
w.add_embedding_length(a.n_embd)
w.add_block_count(a.n_layer)
w.add_feed_forward_length(a.n_ff)
w.add_head_count(a.n_head)
w.add_head_count_kv(a.n_head_kv)
w.add_rope_freq_base(10000.0)
w.add_rope_dimension_count(head_dim)
w.add_layer_norm_rms_eps(1e-5)
w.add_vocab_size(a.vocab)
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

# --- weights --------------------------------------------------------------------------------
w.add_tensor("token_embd.weight", rnd((a.vocab, a.n_embd), 1.0))
for il in range(a.n_layer):
    p = f"blk.{il}."
    w.add_tensor(p + "attn_norm.weight", np.ones(a.n_embd, np.float32))
    w.add_tensor(p + "attn_q.weight", rnd((a.n_embd, a.n_embd)))
    w.add_tensor(p + "attn_k.weight", rnd((n_embd_kv, a.n_embd)))
    w.add_tensor(p + "attn_v.weight", rnd((n_embd_kv, a.n_embd)))
    w.add_tensor(p + "attn_output.weight", rnd((a.n_embd, a.n_embd)))
    w.add_tensor(p + "ffn_norm.weight", np.ones(a.n_embd, np.float32))
    w.add_tensor(p + "ffn_gate.weight", rnd((a.n_ff, a.n_embd)))
    w.add_tensor(p + "ffn_up.weight", rnd((a.n_ff, a.n_embd)))
    w.add_tensor(p + "ffn_down.weight", rnd((a.n_embd, a.n_ff)))
w.add_tensor("output_norm.weight", np.ones(a.n_embd, np.float32))
w.add_tensor("output.weight", rnd((a.vocab, a.n_embd), 0.5))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {a.out}: n_embd={a.n_embd} n_layer={a.n_layer} n_head={a.n_head}/{a.n_head_kv} n_ff={a.n_ff} vocab={a.vocab}")
