"""Calibration-activation collection for the calibrated-INT8 comparator.

The honest comparator for N2/N2'' is CALIBRATED INT8 (per-channel weight
scales derived from real activation statistics), NOT the naive weight-only
INT8 that measured 0.000 on the FT model (see N1v2).  This module collects
per-linear-layer input activations over the frozen eval slice and stores
compact per-channel stats (absmax / mean|.| / count) to disk for the
quantizer and for provenance.

CPU/GPU-light: hooks the model, streams batches, accumulates running
per-channel absmax + count, then saves one .npz per call.  Designed to run
AFTER the queue's GPU lane is free (a few minutes of forward passes).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn


def collect_calibration_stats(
    vla,
    batches: Iterator[dict],
    max_batches: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> dict:
    """Run forward passes with hooks on every nn.Linear, accumulate per-channel
    input absmax.  Returns {layer_name: {"absmax": (in_features,) float32,
    "count": int}}.

    Uses input-operand hooks (forward_pre_hook) so statistics see exactly
    what each Linear sees, including autocast effects.
    """
    stats: dict[str, dict] = {}

    def make_hook(name):
        def hook(module, inp):
            x = inp[0]
            if x.dim() == 3:          # (batch, seq, features)
                x2 = x.reshape(-1, x.shape[-1])
            else:
                x2 = x.reshape(-1, x.shape[-1])
            am = x2.abs().amax(dim=0).float().cpu()   # (in_features,)
            if name not in stats:
                stats[name] = {"absmax": am, "count": int(x2.shape[0])}
            else:
                st = stats[name]
                st["absmax"] = torch.maximum(st["absmax"], am)
                st["count"] += int(x2.shape[0])
        return hook

    hooks = []
    for name, mod in vla.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    try:
        n = 0
        for b in batches:
            if n >= max_batches:
                break
            with torch.inference_mode(), torch.autocast(device, dtype=dtype):
                vla(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                    pixel_values=_to(b["pixel_values"], device, dtype),
                    labels=b["labels"].to(device))
            n += 1
    finally:
        for h in hooks:
            h.remove()

    out = {
        name: {"absmax": st["absmax"].numpy(),
               "count": st["count"]}
        for name, st in stats.items()
    }
    return out


def _to(x, device, dtype):
    if isinstance(x, dict):
        return {k: v.to(device=device, dtype=dtype) for k, v in x.items()}
    return x.to(device=device, dtype=dtype)


def save_stats(stats: dict, path: Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"absmax__{k}": v["absmax"] for k, v in stats.items()},
        **{f"count__{k}": np.int64(v["count"]) for k, v in stats.items()},
    )
    meta_out = {"n_layers": len(stats), "meta": meta or {}}
    Path(str(path) + ".meta.json").write_text(json.dumps(meta_out, indent=2))


def load_stats(path: Path) -> dict:
    z = np.load(Path(path))
    out: dict[str, dict] = {}
    for key in z.files:
        if key.startswith("absmax__"):
            name = key[len("absmax__"):]
            out.setdefault(name, {})
            out[name]["absmax"] = z[key]
        elif key.startswith("count__"):
            name = key[len("count__"):]
            out.setdefault(name, {})
            out[name]["count"] = int(z[key])
    return out
