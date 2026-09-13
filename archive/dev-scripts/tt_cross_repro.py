"""E3 - TT-cross deep-split triage.

Reconstructs the historical crash of ``PythonKernelSet.tt_cross`` on deep
mode splits (d=8) using the real layer-0 q_proj matrix, then instruments
the internal DMRG-cross stages to bisect the failure mode:

  * pivot degeneracy  — maxvol frame matrices rank-deficient => the
    recursive-cross normalization ``C_k = F_k (F_k[I_k])^{-1}`` blows up
    (LinAlgError / non-finite cores),
  * reshape/index bug — dimension mismatches in core reshaping or the
    fused-index oracle (ValueError / silently wrong reconstruction).

Run:
    .venv/Scripts/python.exe -m scripts.tt_cross_repro
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("QICERT_CKPT", "")) if os.environ.get("QICERT_CKPT") \
    else (REPO / "weights" / "ckpt" / "checkpoints" /
          "step-122500-epoch-55-loss=0.0743.pt")

from qicert.kernels import get_backend  # noqa: E402


def load_qproj() -> np.ndarray:
    import torch
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["model"]
    return sd["llm_backbone"]["llm.model.layers.0.self_attn.q_proj.weight"] \
        .detach().float().numpy()


def factor_dims(n: int, d: int) -> tuple[int, ...]:
    """Same semantics as bench/n2_sweep._factor_dims (descending factors)."""
    base = [1] * d
    for p in sorted(_prime_factors(n), reverse=True):
        i = min(range(d), key=lambda j: base[j])
        base[i] *= p
    return tuple(sorted(base, reverse=True))


def _prime_factors(n: int) -> list[int]:
    fs, p = [], 2
    while p * p <= n:
        while n % p == 0:
            fs.append(p)
            n //= p
        p += 1
    if n > 1:
        fs.append(n)
    return fs


def instrumented_cross(K, W, m_dims, n_dims, ranks):
    """Re-run tt_cross stage by stage with diagnostics at each step."""
    d = len(m_dims)
    shape = tuple(int(m_dims[k]) * int(n_dims[k]) for k in range(d))

    def f(fused_idx: int) -> float:
        multi = np.unravel_index(int(fused_idx), shape)
        row = 0
        for k in range(d):
            ik, jk = divmod(int(multi[k]), int(n_dims[k]))
            row = row * int(m_dims[k]) + ik
        col = 0
        for k in range(d):
            _, jk = divmod(int(multi[k]), int(n_dims[k]))
            col = col * int(n_dims[k]) + jk
        return float(W[row, col])

    print(f"[triage] fused shape={shape} prod={np.prod(shape)}")
    # oracle sanity against dense entries
    errs = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        fi = int(rng.integers(0, np.prod(shape)))
        multi = np.unravel_index(fi, shape)
        r = c = 0
        for k in range(d):
            ik, jk = divmod(int(multi[k]), int(n_dims[k]))
            r = r * int(m_dims[k]) + ik
            c = c * int(n_dims[k]) + jk
        errs.append(abs(f(fi) - float(W[r, c])))
    print(f"[triage] oracle max abs err vs dense: {max(errs):.3e}")

    als = K._cross_als
    import types

    orig_maxvol = K._maxvol

    def spy_maxvol(A, tol=1.05):
        A = np.asarray(A, dtype=float)
        sv = np.linalg.svd(A, compute_uv=False)
        smax, smin = float(sv[0]), float(sv[-1])
        cond = smax / max(smin, 1e-300)
        out = orig_maxvol(A, tol=tol)
        print(f"  [maxvol] shape={A.shape} sigma_max={smax:.3e} "
              f"sigma_min={smin:.3e} cond={cond:.3e} finite={np.isfinite(A).all()}")
        return out

    K._maxvol = spy_maxvol  # type: ignore[method-assign]
    try:
        cores = als(f, shape, ranks)
    except Exception:
        print("[triage] _cross_als RAISED:")
        traceback.print_exc()
        return None
    finally:
        K._maxvol = orig_maxvol  # type: ignore[method-assign]
    norms = [float(np.linalg.norm(c.reshape(c.shape[0] * c.shape[1], -1), 2))
             for c in cores]
    print(f"[triage] cores finite={fin} norms={['%.3e' % x for x in norms]}")
    return cores


def main() -> int:
    K = get_backend("python")
    W = load_qproj()
    M, N = W.shape
    print(f"[triage] q_proj {M}x{N}")

    for d in (2, 4, 8):
        m_dims = factor_dims(M, d)
        n_dims = factor_dims(N, d)
        ranks = tuple([8] * (d - 1))
        print(f"\n=== tt_cross d={d} m={m_dims} n={n_dims} r={ranks} ===")
        try:
            cs = K.tt_cross(W, m_dims, n_dims, ranks)
            Wr = K.contract_cores(cs.arrays, m_dims, n_dims)
            rel = float(np.linalg.norm(Wr - W) / np.linalg.norm(W))
            fin = bool(np.all([np.isfinite(g).all() for g in cs.arrays]))
            print(f"[triage] d={d}: recon rel err={rel:.4f} finite={fin}")
        except Exception as exc:
            print(f"[triage] d={d}: CRASHED {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print("\n--- instrumented re-run ---")
            instrumented_cross(K, W, m_dims, n_dims, ranks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
