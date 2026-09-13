"""Calibration-activation collection for the calibrated-INT8 comparator.

The honest classical comparator is CALIBRATED INT8 (per-channel weight
scales derived from real activation statistics), NOT the naive weight-only
INT8 that scored 0.000 on the fine-tuned model (a calibration artifact).
This module collects per-linear-layer input activations over train episodes
and stores compact per-channel stats (absmax / mean|.| / count) to disk
for the quantizer and for provenance.

CPU/GPU-light: hooks the model, streams batches, accumulates running
per-channel absmax + count, then saves one .npz per call (a few minutes of
forward passes; run while no training job holds the GPU).
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
            sm = x2.abs().sum(dim=0).float().cpu()    # (in_features,) for mean|.| 
            if name not in stats:
                stats[name] = {"absmax": am, "abssum": sm,
                               "count": int(x2.shape[0])}
            else:
                st = stats[name]
                st["absmax"] = torch.maximum(st["absmax"], am)
                st["abssum"] = st["abssum"] + sm
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
               "mean_abs": (st["abssum"].numpy()
                            / max(st["count"], 1)).astype(np.float32),
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
        **{f"mean_abs__{k}": v["mean_abs"] for k, v in stats.items()},
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
        elif key.startswith("mean_abs__"):
            name = key[len("mean_abs__"):]
            out.setdefault(name, {})
            out[name]["mean_abs"] = z[key]
        elif key.startswith("count__"):
            name = key[len("count__"):]
            out.setdefault(name, {})
            out[name]["count"] = int(z[key])
    return out


# ---------------------------------------------------------------------------
# CLI: collect calibration stats over the LOCAL NPZ bridge (train episodes).
#
# Audit finding C (2026-09-12): the first calibration attempt went through
# the fork's RLDS pipeline and failed with "No registered data_dirs" — a
# wrong-loader artifact, NOT a missing dataset.  Every working bench run
# (N1/N2/N2R) uses qicert.data.npz_loader + vla_local (the local bridge).
# Calibration must too.  Methodology: calibrate on TRAIN episodes from the
# frozen split (never on the held-out eval episodes).
# ---------------------------------------------------------------------------
def _main() -> int:
    import argparse
    import sys
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="collect per-channel calibration stats")
    ap.add_argument("--ckpt", default=str(repo / "weights" / "ckpt" / "checkpoints"
                                           / "step-122500-epoch-55-loss=0.0743.pt"))
    ap.add_argument("--ft-llm", default="",
                    help="optional FT llm_backbone handoff (e.g. "
                         "results/N1v2-ckpt/seed1.pt): installs its "
                         "llm_backbone weights onto the base VLA before "
                         "collecting, so stats reflect the fine-tuned model. "
                         "Empty = base weights (seed-0 convention).")
    ap.add_argument("--npz-root", default=str(repo / "weights"
                                               / "libero_spatial_no_noops_npz"))
    ap.add_argument("--split", default=str(repo / "results" / "eval_split.json"))
    ap.add_argument("--out", default=str(repo / "results" / "N2R"
                                         / "calib_seed0.npz"))
    ap.add_argument("--batches", type=int, default=256,
                    help="number of calibration batches (train side)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import json
    import os
    import torch

    os.environ.setdefault("PRISMATIC_DATA_ROOT", str(repo / "weights" / "data"))
    sys.path.insert(0, str(repo / "weights" / "code"))
    from qicert.transformers5_compat import install
    install()
    from prismatic.models.load import load_vla

    from qicert.data.npz_loader import NpzEpisodeDataset
    from qicert.data.vla_local import LocalEvalSplitStream, LocalVLABatcher

    print(f"[calibrate] loading VLA from {args.ckpt}", flush=True)
    vla = load_vla(args.ckpt, hf_token=None, load_for_training=False)
    vla = vla.to(dtype=torch.float16, device="cuda")
    if args.ft_llm:
        ft = torch.load(args.ft_llm, map_location="cpu", weights_only=True)
        ft_llm = ft["llm_backbone"] if "llm_backbone" in ft else ft
        vla.llm_backbone.load_state_dict(ft_llm, strict=False)
        del ft, ft_llm
        print(f"[calibrate] installed FT llm weights from {args.ft_llm}",
              flush=True)
    vla.llm_backbone.eval()

    ep_ds = NpzEpisodeDataset(args.npz_root)
    lb = LocalVLABatcher(vla)

    # Train episodes only (honest calibration: never touch held-out data).
    # The split file stores eval names + a train COUNT, so train = all files
    # minus the eval set.  LocalEvalSplitStream is used (finite, ordered)
    # rather than LocalEpisodeStream (infinite, all episodes incl. eval).
    split = json.loads(_Path(args.split).read_text())
    eval_set = set(split["eval"])
    train_eps = [p.name for p in ep_ds.files if p.name not in eval_set]
    print(f"[calibrate] {len(train_eps)} train episodes (all-minus-eval) -> "
          f"{args.batches} batches of {args.batch_size}", flush=True)

    import itertools
    stream = itertools.islice(
        LocalEvalSplitStream(ep_ds, lb, train_eps, batch_size=args.batch_size),
        args.batches)
    stats = collect_calibration_stats(vla, stream, max_batches=args.batches,
                                      device="cuda", dtype=torch.float16)
    save_stats(stats, _Path(args.out), meta={
        "ckpt": args.ckpt,
        "ft_llm": args.ft_llm or "base (no FT weights installed)",
        "npz_root": args.npz_root,
        "split": args.split,
        "episodes_scope": "train (from results/eval_split.json)",
        "n_batches": args.batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_layers": len(stats),
    })
    print(f"[calibrate] wrote {args.out} ({len(stats)} linear layers)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
