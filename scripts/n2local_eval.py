"""E10b - Reduced-N2 accuracy points measured LOCALLY (N2local).

Compresses the saved N1local fine-tuned checkpoint via TT-SVD at target
param fractions {0.01, 0.02, 0.04} (uniform-ratio allocator, TT d=2
bit-reversed - the N2/N2pre convention) and scores each compressed model on
the enlarged held-out protocol built from the NPZ bridge (E8): 24 episodes x
4 spread frames, teacher-forced action-token accuracy with Wilson 95% CI.

Preprocessing uses qicert.data.vla_local.LocalVLABatcher - the fork-equivalent
transform/collator semantics without dlimp/tensorflow (documented
substitution; AGENTS.md).

Run:
    .venv/Scripts/python.exe -m scripts.n2local_eval \
        --ckpt results/N1local-ckpt/seed0.pt --out results --exp-id N2local
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FORK = REPO / "weights" / "code"
CKPT_BASE = REPO / "weights" / "ckpt" / "checkpoints" / \
    "step-122500-epoch-55-loss=0.0743.pt"
NPZ_ROOT = REPO / "weights" / "libero_spatial_no_noops_npz"

FRACTIONS = (0.01, 0.02, 0.04)


def _wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.n2local_eval")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--frames-per-episode", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exp-id", default="N2local")
    args = ap.parse_args()

    os.environ.setdefault("PRISMATIC_DATA_ROOT", str(REPO / "weights" / "data"))
    if str(FORK) not in sys.path:
        sys.path.insert(0, str(FORK))
    sentinel = FORK / "prismatic" / "models" / "backbones" / "llm" / \
        ".qicert_patched"
    assert sentinel.exists(), "fork missing transformers-5.x patch"

    import torch

    from bench.n2_sweep import _factor_dims, _rank_for_fraction
    from prismatic.models.load import load_vla

    from qicert.data.npz_loader import NpzEpisodeDataset
    from qicert.data.vla_local import LocalVLABatcher
    from qicert.kernels import get_backend

    k = get_backend()
    fracs = tuple(float(x) for x in args.fractions.split(",") if x.strip())

    print("[n2local] loading MiniVLA (fp16)...", flush=True)
    vla = load_vla(str(CKPT_BASE), hf_token=None, load_for_training=False)
    vla = vla.to(dtype=torch.float16, device="cuda").eval()
    vla._optimize_model_for_decode = \
        lambda: __import__("contextlib").nullcontext()
    num_patches = int(vla.vision_backbone.num_patches)

    lb = LocalVLABatcher(vla)
    begin_idx = lb.action_tokenizer.action_token_begin_idx

    # ---- build the enlarged eval set from the NPZ bridge ----
    ds = NpzEpisodeDataset(NPZ_ROOT)
    assert len(ds) >= args.episodes, \
        f"need {args.episodes} episodes, bridge has {len(ds)}"
    batches = []
    n_samples = 0
    for e in range(args.episodes):
        obs, acts, lang = ds[e]
        T = len(obs)
        picks = np.linspace(0, T - 1, args.frames_per_episode).astype(int)
        examples = []
        for t in picks:
            examples.append(lb.example(obs[t], acts[t], lang))
            n_samples += 1
            if len(examples) == args.eval_batch:
                batches.append(lb.collate(examples))
                examples = []
    if examples:
        batches.append(lb.collate(examples))
    print(f"[n2local] eval set: {n_samples} samples in {len(batches)} batches "
          f"({args.episodes} episodes x {args.frames_per_episode} frames)",
          flush=True)

    def score() -> tuple[int, int]:
        correct = total = 0
        with torch.inference_mode():
            for b in batches:
                pv = b["pixel_values"]
                if isinstance(pv, dict):
                    pv = {kk: vv.to(torch.float16).cuda()
                          for kk, vv in pv.items()}
                else:
                    pv = pv.to(torch.float16).cuda()
                out_ = vla(input_ids=b["input_ids"].cuda(),
                           attention_mask=b["attention_mask"].cuda(),
                           pixel_values=pv,
                           labels=b["labels"].cuda())
                logits = out_.logits[:, num_patches:-1]
                preds = logits.argmax(dim=-1)
                gt = b["labels"][:, 1:].cuda()
                mask = gt > begin_idx
                correct += int((preds[mask] == gt[mask]).sum().item())
                total += int(mask.sum().item())
        return correct, total

    def record(label: str, frac, c: int, t: int, ratio: float, ttp: int):
        lo, hi = _wilson(c, t)
        acc = c / max(t, 1)
        print(f"[n2local] {label}: acc={acc:.4f} [{lo:.4f},{hi:.4f}] "
              f"n={t}", flush=True)
        if not args.out:
            return
        from qicert.record import RunRecorder
        rec = RunRecorder(exp_id=args.exp_id, seed=0,
                          config={"experiment": "E10b local reduced-N2",
                                  "arm": label, "fraction": frac,
                                  "tt_ratio": round(ratio, 3),
                                  "tt_params": ttp,
                                  "ckpt": str(args.ckpt),
                                  "protocol": {
                                      "episodes": args.episodes,
                                      "frames_per_episode":
                                          args.frames_per_episode,
                                      "batch": args.eval_batch}},
                          out_root=args.out, run_tag=label)
        rec.metric(correct=c, total=t, action_acc=float(acc),
                   ci_lo=float(lo), ci_hi=float(hi))
        rec.sample_power()
        rec.finalize(status="completed",
                     results={"action_acc": float(acc),
                              "ci95": [round(lo, 5), round(hi, 5)],
                              "n_tokens": t, "arm": label},
                     tolerance_note="teacher-forced action-token accuracy "
                                    "over NPZ-bridge enlarged holdout; "
                                    "fork-equivalent local preprocessing")

    # ---- fine-tuned merged backbone ----
    ft = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    merged = ft["llm_backbone"]

    inv_pat = re.compile(r"^llm\.model\.layers\.\d+\."
                         r"(self_attn\.(?:q|k|v|o)_proj|"
                         r"mlp\.(?:gate|up|down)_proj)\.weight$")
    inventory = sorted(kk for kk in merged if inv_pat.match(kk))

    def compress_merged(fraction: float) -> tuple[dict, int, int]:
        comp = {}
        tt_params = dense_params = 0
        for kk in inventory:
            W = merged[kk].detach().float().cpu().numpy()
            M, N = W.shape
            m_dims = _factor_dims(M, 2)
            n_dims = _factor_dims(N, 2)
            r = _rank_for_fraction(fraction, m_dims, n_dims)
            cs = k.tt_svd(W, m_dims, n_dims, (r,))
            Wr = k.contract_cores(cs.arrays, m_dims, n_dims)
            comp[kk] = torch.from_numpy(Wr)
            tt_params += int(sum(np.prod(g.shape) for g in cs.arrays))
            dense_params += M * N
        return comp, tt_params, dense_params

    vla.llm_backbone.load_state_dict(merged, strict=False)
    c, t = score()
    record("ft-baseline", None, c, t, 1.0,
           sum(int(merged[kk].numel()) for kk in inventory))

    for frac in fracs:
        comp, ttp, dnp = compress_merged(frac)
        vla.llm_backbone.load_state_dict(comp, strict=False)
        c, t = score()
        ratio = dnp / max(ttp, 1)
        record(f"frac{frac:.3f}", frac, c, t, ratio, ttp)

    print("\n* N2local complete: first LOCAL accuracy-vs-compression points "
          "(ledger rows tagged N2local); comparable to the E7 INT8 arms at "
          "the matched eval protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
