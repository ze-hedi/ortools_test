#!/usr/bin/env python3
"""Read SmolLM3-3B safetensors and print every tensor one by one."""

import json
from pathlib import Path
from safetensors import safe_open

SNAPSHOT = next(
    (Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B/snapshots").iterdir()
)

# Load index to know which shard has which tensor
index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())

# Open both shards
shards = {}
for shard_name in set(index["weight_map"].values()):
    shards[shard_name] = safe_open(str(SNAPSHOT / shard_name), framework="pt")

# Walk through every tensor
for name in sorted(index["weight_map"]):
    shard = index["weight_map"][name]
    tensor = shards[shard].get_tensor(name)
    size_mb = tensor.nelement() * 2 / 1e6  # bf16 = 2 bytes
    print(f"{name}")
    print(f"  shape: {list(tensor.shape)}  dtype: {tensor.dtype}  size: {size_mb:.1f} MB")
    print()
