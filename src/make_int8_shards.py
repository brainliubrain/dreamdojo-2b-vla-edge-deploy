"""One-time converter: bf16 transformer shards -> INT8 quantized shards.

Processes one tensor at a time (peak RAM ~700MB) so it works under tight
system commit limits. Output keys:
  - quantized weights: original key -> int8 tensor, key + ".scales" -> bf16 (out,1)
  - everything else: original key -> bf16 unchanged
Only keys matching Int8WeightLinear targets are quantized, matching
dreamdojo_standalone_engine_win.Int8WeightLinear.set_weight exactly.
"""
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "D:/DreamDojo_2B/transformer"
DST = "D:/DreamDojo_2B/transformer_int8"
SHARD_BUDGET = 150 * 1024 * 1024  # small: system commit is tight (WSL VM holds ~6GB)

Q_PAT = re.compile(
    r"^blocks\.\d+\.(?:self_attn|cross_attn)\.(?:q_proj|k_proj|v_proj|output_proj)\.weight$"
    r"|^blocks\.\d+\.mlp\.layer[12]\.weight$")


def quantize_rowwise(w_bf16):
    max_abs = w_bf16.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scale = max_abs / 127.0
    q = (w_bf16 / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


def main():
    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(SRC, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    # group keys by source shard to stream sequentially
    by_shard = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)

    out_shard_id = 0
    out_tensors = {}
    out_bytes = 0
    index = {}
    n_quant = n_copy = 0

    def flush():
        nonlocal out_shard_id, out_tensors, out_bytes
        if not out_tensors:
            return
        name = f"model-int8-{out_shard_id:05d}.safetensors"
        save_file(out_tensors, os.path.join(DST, name),
                  metadata={"format": "pt"})
        for k in out_tensors:
            index[k] = name
        print(f"  wrote {name} ({out_bytes/1e6:.0f} MB, {len(out_tensors)} tensors)", flush=True)
        out_shard_id += 1
        out_tensors = {}
        out_bytes = 0

    for shard in sorted(by_shard):
        print(f"reading {shard} ...", flush=True)
        with safe_open(os.path.join(SRC, shard), framework="pt", device="cpu") as f:
            for k in by_shard[shard]:
                t = f.get_tensor(k).to(torch.bfloat16)
                if Q_PAT.match(k):
                    q, s = quantize_rowwise(t)
                    out_tensors[k] = q.contiguous()
                    out_tensors[k + ".scales"] = s.contiguous()
                    out_bytes += q.numel() + s.numel() * 2
                    n_quant += 1
                else:
                    out_tensors[k] = t.contiguous()
                    out_bytes += t.numel() * 2
                    n_copy += 1
                del t
                if out_bytes >= SHARD_BUDGET:
                    flush()
    flush()

    with open(os.path.join(DST, "model_int8_index.json"), "w") as f:
        json.dump({"weight_map": index, "metadata": {"format": "pt"}}, f)
    print(f"DONE: {n_quant} quantized, {n_copy} copied, {len(index)} keys, "
          f"{out_shard_id} shards -> {DST}", flush=True)


if __name__ == "__main__":
    main()
