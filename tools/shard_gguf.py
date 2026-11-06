#!/usr/bin/env python3
"""Cut a GGUF into per-layer shards plus a manifest, so a node can fetch only its layers.

    python tools/shard_gguf.py model.gguf --out shards/ [--name NAME] [--seed URL ...]

Layout (GGUF split files, loadable by llama.cpp as one model):
    NAME-00001-of-000NN.gguf   all metadata plus small global tensors (norms, rope tables)
    NAME-00002-of-000NN.gguf   the token embedding (and other input-side tensors)
    NAME-0000k-of-000NN.gguf   blk.(k-3).* : one shard per transformer layer
    NAME-000NN-of-000NN.gguf   the output head, when it is not tied to the embedding

A stage for layers [a, b) needs the first shard and the shards for its layers; the
loader (with the dfin layer-range patch) skips shards that are not on disk. The
manifest NAME.manifest.json lists every shard with its size and SHA-256, and the
model id is a hash over those hashes, so every holder of the same shard set agrees
on what model it is without holding all of it.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402

FORMAT = "dfin-shards-v1"
BLK = re.compile(r"^blk\.(\d+)\.")
INPUT_SIDE = ("token_embd", "token_types", "position_embd", "token_embd_norm")
OUTPUT_SIDE = ("output.weight", "output.bias")


def kind_of(name: str):
    m = BLK.match(name)
    if m:
        return "layer", int(m.group(1))
    if name.startswith(INPUT_SIDE):
        return "embd", None
    if name in OUTPUT_SIDE:
        return "output", None
    return "meta", None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_id(splits) -> str:
    h = hashlib.sha256()
    h.update((FORMAT + "\n").encode())
    for s in splits:
        h.update(f"{s['no']}:{s['sha256']}\n".encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", help="shard file prefix (default: the model file stem)")
    ap.add_argument("--seed", action="append", default=[], help="HTTP URL prefix where the shards will be served")
    args = ap.parse_args()

    src = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or src.stem
    reader = GGUFReader(str(src))
    arch = reader.fields["general.architecture"].contents()
    n_layer = int(reader.fields[f"{arch}.block_count"].contents())

    groups = {"meta": [], "embd": [], "output": []}
    layers = {i: [] for i in range(n_layer)}
    for t in reader.tensors:
        kind, il = kind_of(t.name)
        if kind == "layer":
            if il not in layers:
                sys.exit(f"tensor {t.name} names layer {il} beyond block_count {n_layer}")
            layers[il].append(t)
        else:
            groups[kind].append(t)

    plan = [("meta", None, groups["meta"]), ("embd", None, groups["embd"])]
    plan += [("layer", i, layers[i]) for i in range(n_layer)]
    if groups["output"]:
        plan.append(("output", None, groups["output"]))
    n_split = len(plan)
    total = sum(len(ts) for _, _, ts in plan)

    splits = []
    for no, (kind, il, tensors) in enumerate(plan):
        fname = f"{name}-{no + 1:05d}-of-{n_split:05d}.gguf"
        path = out / fname
        w = GGUFWriter(str(path), arch)
        if no == 0:
            for field in reader.fields.values():
                if field.name.startswith("GGUF.") or field.name == "general.architecture" or field.name.startswith("split."):
                    continue
                vtype = field.types[0]
                sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
                w.add_key_value(field.name, field.contents(), vtype, sub_type=sub)
        w.add_uint16(gguf.Keys.Split.LLM_KV_SPLIT_NO, no)
        w.add_uint16(gguf.Keys.Split.LLM_KV_SPLIT_COUNT, n_split)
        w.add_int32(gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT, total)
        for t in tensors:
            # The reader hands back the data in numpy order: element shape for plain types,
            # byte shape for quantized ones, which is what the writer expects for each.
            w.add_tensor_info(t.name, list(t.data.shape), t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        for t in tensors:
            w.write_tensor_data(t.data)
        w.close()
        splits.append({
            "no": no,
            "file": fname,
            "size": os.path.getsize(path),
            "sha256": sha256_file(path),
            "kind": kind,
            "layer": il,
            "tensors": [t.name for t in tensors],
        })
        print(f"{fname}: {kind}{'' if il is None else ' ' + str(il)}, {len(tensors)} tensors, {splits[-1]['size']} bytes")

    manifest = {
        "format": FORMAT,
        "model_name": name,
        "arch": arch,
        "n_layer": n_layer,
        "first": splits[0]["file"],
        "splits": splits,
        "seeds": args.seed,
    }
    manifest["model_id"] = model_id(splits)
    mpath = out / f"{name}.manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest {mpath}: model id {manifest['model_id']}, {n_split} shards, {total} tensors")


if __name__ == "__main__":
    main()
