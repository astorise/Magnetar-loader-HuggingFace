"""Generate E4M3FN and partial-block references with PyTorch 2.15.0.

The block operation follows DeepSeek-V3's published weight_dequant kernel:
each decoded FP8 value is multiplied by the scale for its row/column tile.
No model checkpoint weights are included.
"""

import json
import math
from pathlib import Path
import struct

import torch

if not torch.__version__.startswith("2.15.0"):
    raise RuntimeError("this reference was generated with PyTorch 2.15.0")


def bits(value):
    value = float(value)
    return None if math.isnan(value) else struct.unpack("<I", struct.pack("<f", value))[0]


raw = torch.arange(256, dtype=torch.int64).to(torch.uint8)
decoded = raw.view(torch.float8_e4m3fn).to(torch.float32)
weight = [(index * 37 + 1) % 256 for index in range(35)]
scales = [0.5, 1.0, 2.0, 0.0, -1.0, 4.0, 0.125, 3.0, 8.0]
matrix = torch.tensor(weight, dtype=torch.uint8).view(torch.float8_e4m3fn).float().reshape(5, 7)
scale_grid = torch.tensor(scales, dtype=torch.float32).reshape(3, 3)
expanded = scale_grid.repeat_interleave(2, dim=0).repeat_interleave(3, dim=1)[:5, :7]
result = matrix * expanded
fixture = {
    "decoded_bits": [bits(value) for value in decoded],
    "weight": weight,
    "scales": scales,
    "scaled_bits": [bits(value) for value in result.flatten()],
}
Path(__file__).with_name("e4m3fn-reference.json").write_text(
    json.dumps(fixture, indent=2) + "\n", encoding="utf-8"
)
