"""E2 - Layer-spectra profiler for the MiniVLA LLM backbone.

Loads the checkpoint state dict, iterates every transformer layer x
{self_attn q/k/v/o_proj, mlp gate/up/down_proj}, computes the singular-value
spectrum on GPU, and derives the three compressibility statistics that feed
the N2 bond allocator and the report figure:

  * participation ratio  PR = (sum s)^2 / sum(s^2)
  * stable rank          SR = ||W||_F^2 / ||W||_2^2 = sum(s^2)/s_max^2
  * decay slope          fit of log(s_i) vs index i over the top-64 values

Outputs results/layer_spectra.csv plus a stdout markdown table grouped by
projection type. No dataset needed; GPU optional (falls back to CPU).

Run:
    .venv/Scripts/python.exe -m scripts.layer_spectra [--out results]
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

# Same resolution pattern as bench/compress.py::CKPT (env override first).
REPO = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("QICERT_CKPT", "")) if os.environ.get("QICERT_CKPT") \
    else REPO / "weights" / "ckpt" / "checkpoints" / \
    "step-122500-epoch-55-loss=0.0743.pt"

PROJECTIONS = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}
TOP_SLOPE = 64


def _layer_keys(sd: dict) -> dict[int, dict[str, str]]:
    """depth -> {suffix -> state-dict key} for llm.model.layers.N.* weights."""
    found: dict[int, dict[str, str]] = {}
    for key in sd:
        if not key.startswith("llm.model.layers."):
            continue
        parts = key.split(".")
        try:
            depth = int(parts[3])
        except ValueError:
            continue
        suffix = ".".join(parts[4:])
        found.setdefault(depth, {})[suffix] = key
    return found

def _spectra(W: np.ndarray, dev) -> dict:
    import torch

    t = torch.from_numpy(W).to(device=dev, dtype=torch.float32)
    s = torch.linalg.svdvals(t).cpu().numpy()
    top = s[:min(TOP_SLOPE, len(s))]
    idx = np.arange(len(top), dtype=float)
    slope = float(np.polyfit(idx, np.log(np.maximum(top, 1e-300)), 1)[0])
    return {
        "sigma1": float(s[0]),
        "participation_ratio": float(s.sum() ** 2 / np.sum(s ** 2)),
        "stable_rank": float(np.sum(s ** 2) / s[0] ** 2),
        "decay_slope_top64": slope,
    }


def main() -> int:
    import torch  # torch-optional module: local import

    ap = argparse.ArgumentParser(prog="scripts.layer_spectra")
    ap.add_argument("--out", default=str(REPO / "results"),
                    help="output dir for layer_spectra.csv")
    ap.add_argument("--cpu", action="store_true", help="force CPU SVD")
    args = ap.parse_args()

    if not CKPT.exists():
        print(f"checkpoint not found: {CKPT} "
              "(run scripts/download_backbones.py first)", flush=True)
        return 1
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                       else "cuda")
    print(f"[layer_spectra] ckpt={CKPT.name} device={dev}", flush=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["model"]
    llm = sd["llm_backbone"] if "llm_backbone" in sd else sd
    layers = _layer_keys(llm)
    depths = sorted(layers)
    print(f"[layer_spectra] {len(depths)} transformer layers x "
          f"{sum(len(p) for p in PROJECTIONS.values())} projections",
          flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "layer_spectra.csv"
    cols = ["layer", "proj", "M", "N", "params", "sigma1",
            "participation_ratio", "stable_rank", "decay_slope_top64"]

    t0 = time.perf_counter()
    rows: list[dict] = []
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for depth in depths:
            for group, projs in PROJECTIONS.items():
                for p in projs:
                    key = layers[depth].get(f"{group}.{p}.weight")
                    if key is None:
                        continue
                    W = llm[key].detach().float().cpu().numpy()
                    st = _spectra(W, dev)
                    row = {"layer": depth, "proj": f"{group}.{p}",
                           "M": W.shape[0], "N": W.shape[1],
                           "params": int(W.size), **st}
                    rows.append(row)
                    wr.writerow(row)
            if depth % 8 == 0 or depth == depths[-1]:
                print(f"  layer {depth}: done ({time.perf_counter() - t0:.1f}s)",
                      flush=True)
    fh_elapsed = time.perf_counter() - t0

    # Markdown summary grouped by projection type (mean +- std over layers).
    print("\n### Layer spectra summary (mean ± std across "
          f"{len(depths)} layers)\n")
    print("| Projection | M x N | Part. ratio | Stable rank | "
          "Slope (top-64) |")
    print("|---|---|---|---|---|")
    projs = [f"{g}.{p}" for g, ps in PROJECTIONS.items() for p in ps]
    for proj in projs:
        sel = [r for r in rows if r["proj"] == proj]
        if not sel:
            continue
        pr = np.array([r["participation_ratio"] for r in sel])
        sr = np.array([r["stable_rank"] for r in sel])
        sl = np.array([r["decay_slope_top64"] for r in sel])
        shape = f"{sel[0]['M']}x{sel[0]['N']}"
        print(f"| {proj} | {shape} | {pr.mean():.2f} ± {pr.std():.2f} | "
              f"{sr.mean():.2f} ± {sr.std():.2f} | "
              f"{sl.mean():.3f} ± {sl.std():.3f} |")
    print(f"\nwrote {csv_path} ({len(rows)} rows, {fh_elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
