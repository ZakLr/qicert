"""E5 - P1 residual pilot: INT4 base + certified low-rank/TT residual?

Per selected layers (q_proj/o_proj/up_proj at depths {0, 15}):
  1. simulate per-channel symmetric INT4 (scale = max|w|_ch / 7,
     round-to-nearest) -> quantization residual R = W - Q(W);
  2. fit corrections at MATCHED parameter budget b:
       - dense rank-b' SVD of R  (params ~ b'(M+N))
       - TT-SVD of R (d=2, bit-reversed) at equal param count
  3. report two objectives per arm:
       - plain LS (weight space): ||R - fit||_F
       - activation-weighted: sqrt(trace(D^T C D)) with C the cached input
         Gram from E7 hooks (results/act_stats.npz) if available, else the
         identity (printed warning) -> equals ||D||_F.

Decision this feeds: does the spine become "INT4 base + certified QTT
residual" (TT residual close to dense-SVD residual at equal budget) or not.

Run:
    .venv/Scripts/python.exe -m scripts.residual_pilot [--out results]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("QICERT_CKPT", "")) if os.environ.get("QICERT_CKPT") \
    else (REPO / "weights" / "ckpt" / "checkpoints" /
          "step-122500-epoch-55-loss=0.0743.pt")
STATS = REPO / "results" / "act_stats.npz"

LAYERS = [("q_proj", 0), ("o_proj", 0), ("up_proj", 0),
          ("q_proj", 15), ("o_proj", 15), ("up_proj", 15)]
RADICES_BUDGET = (8, 16, 32)


def factorize(n: int) -> tuple[int, int]:
    a = int(round(n ** 0.5))
    while n % a != 0:
        a -= 1
    return (a, n // a)


def int4_quant_perchannel(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel symmetric INT4: s_c = max|w_c| / 7, round-to-nearest."""
    scale = np.abs(W).max(axis=1, keepdims=True) / 7.0
    scale = np.maximum(scale, 1e-12)
    Q = np.clip(np.round(W / scale), -7, 7)
    return Q * scale, scale


def tt_svd_d2(W: np.ndarray, bond: int):
    from qicert.kernels import get_backend

    M, N = W.shape
    ma, mb = factorize(M)
    na, nb = factorize(N)
    m_dims = tuple(sorted((ma, mb), reverse=True))
    n_dims = tuple(sorted((na, nb), reverse=True))
    cs = get_backend().tt_svd(W, m_dims, n_dims, (bond,))
    Wr = get_backend().contract_cores(cs.arrays, m_dims, n_dims)
    params = int(sum(np.prod(g.shape) for g in cs.arrays))
    return Wr, params


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(prog="scripts.residual_pilot")
    ap.add_argument("--out", default=None, help="results root (recorder)")
    args = ap.parse_args()

    rec = None
    if args.out:
        from qicert.record import RunRecorder
        rec = RunRecorder(exp_id="E5-residual-pilot", seed=0,
                          config={"experiment": "P1 residual pilot",
                                  "int4": "per-channel symmetric, s=max|w|/7",
                                  "budgets": list(RADICES_BUDGET),
                                  "layers": [f"{p}@d{d}" for p, d in LAYERS],
                                  "activation_weighting":
                                      "act_stats.npz if present else identity"},
                          out_root=args.out, run_tag="int4-residual")

    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["model"]
    llm = sd["llm_backbone"]

    # Activation Grams from E7 hooks (optional).
    grams: dict[str, np.ndarray] = {}
    if STATS.exists():
        z = np.load(STATS)
        grams = {k: z[k].astype(np.float64) for k in z.files}
        print(f"[residual] loaded activation Grams: {sorted(grams)}")
    else:
        print("[residual] WARNING: results/act_stats.npz missing (run "
              "scripts/int8_ablation.py first) -> identity covariance "
              "fallback; weighted column == Frobenius column.", flush=True)

    print("\n### P1 residual pilot - INT4 base + matched-budget correction\n")
    print("| Layer | M x N | Budget b | ||R||_F | dense LS resid | "
          "TT LS resid | dense w-resid | TT w-resid | TT/dense w | Params |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    worst_ratio = 0.0
    for proj, depth in LAYERS:
        key = f"llm.model.layers.{depth}.self_attn.{proj}.weight"
        if f".{proj}." not in key or key not in llm:
            alt = [k for k in llm if f".layers.{depth}." in k
                   and k.endswith(f"{proj}.weight")]
            key = alt[0]
        W = llm[key].detach().float().cpu().numpy().astype(np.float64)
        M, N = W.shape
        Q, _ = int4_quant_perchannel(W)
        R = W - Q
        fro_R = float(np.linalg.norm(R))
        C = grams.get(key)
        if C is None:
            C = np.eye(min(M, N))  # placeholder identity (see warning)

        def w_resid(D: np.ndarray) -> float:
            # E||Delta y||^2 = trace(D^T C D); C lives on the input side (N)
            if C.shape == (N, N):
                return float(np.sqrt(np.einsum("ij,jk,ik->", D, C, D)))
            return float(np.linalg.norm(D))

        for b in RADICES_BUDGET:
            budget = b * (M + N)
            U, S, Vt = np.linalg.svd(R, full_matrices=False)
            bd = min(b, len(S))
            D_dense = (U[:, :bd] * S[:bd]) @ Vt[:bd]
            # TT at equal param count: bond solves bond*(m1*n1 + m2*n2) = b(M+N)
            ma, mb_ = factorize(M)
            na, nb = factorize(N)
            m_dims = sorted((ma, mb_), reverse=True)
            n_dims = sorted((na, nb), reverse=True)
            denom = m_dims[0] * n_dims[0] + m_dims[1] * n_dims[1]
            bt = max(1, int(round(budget / denom)))
            D_tt, p_tt = tt_svd_d2(R, bt)
            res_dense_ls = float(np.linalg.norm(R - D_dense))
            res_tt_ls = float(np.linalg.norm(R - D_tt))
            res_dense_w = w_resid(R - D_dense)
            res_tt_w = w_resid(R - D_tt)
            ratio = res_tt_w / max(res_dense_w, 1e-300)
            worst_ratio = max(worst_ratio, ratio)
            print(f"| {proj}@d{depth} | {M}x{N} | {b} | {fro_R:.3f} | "
                  f"{res_dense_ls:.3f} | {res_tt_ls:.3f} | {res_dense_w:.3f} | "
                  f"{res_tt_w:.3f} | {ratio:.2f} | "
                  f"{b * (M + N)}/{p_tt} |")
            if rec:
                rec.metric(layer=f"{proj}@{depth}", budget=b,
                           frobenius_residual_R=fro_R,
                           dense_ls_residual=res_dense_ls,
                           tt_ls_residual=res_tt_ls,
                           dense_weighted_residual=res_dense_w,
                           tt_weighted_residual=res_tt_w,
                           tt_over_dense_weighted=round(ratio, 4),
                           dense_params=b * (M + N), tt_params=p_tt)
    print(f"\n* VERDICT: worst TT/dense weighted-residual ratio = "
          f"{worst_ratio:.2f}. Ratio near 1 => 'INT4 base + certified TT "
          f"residual' is budget-competitive with dense SVD; ratio >> 1 "
          f"=> keep corrections dense.")
    if rec:
        rec.sample_power()
        rec.finalize(status="completed",
                     results={"worst_tt_over_dense_ratio":
                              round(float(worst_ratio), 4)},
                     tolerance_note="matched-parameter comparison; weighted "
                                    "objective uses E7 input Grams when "
                                    "available (else identity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
