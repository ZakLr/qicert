"""Residual compensation for TT-compressed linear layers (N2'' repair arm).

Why this exists (2026-09-12, post go/no-go): uniform-ratio TT-SVD truncation
at 2x/3x compression measured eval acc 0.000 (down from FT ref 0.447) — the
uniform allocator removes signal, not redundancy, on this backbone's flat
spectra.  The pre-registered repair is closed-form residual compensation
(QuaSAR-style, arXiv:2608.14149, adopted per lit-swarm L1/L5 findings):

    keep the TT reconstruction  What = TT(W)
    compute what was lost       R = W - What
    fold back, cheaply:         W_comp = What + S_out @ (U V^T) @ S_in

where U V^T is the best rank-r' approximation of R (r' << min(M, N)) and
S_in / S_out are diagonal per-channel scales.  The budget accounting is
honest: the residual factors and scales are stored alongside the TT cores,
so the reported compression ratio includes them — the comparison against
calibrated INT8 happens at MATCHED total parameter count.

Certificate note (the differentiator): the certified operator is the pair
    y = TT_matvec(x) + U (V^T (S_in x))
which is a linear composition — its Lipschitz bound is the product of the
TT layer bound, ||U||_2, ||V^T||_2, and the diagonal scales' operator norm
(max |s|).  All four factors are computable; `lipschitz_bound` returns it.
The bound is SOUND (upper bound on the true operator norm), which is what
the safe-set chain requires.
"""
from __future__ import annotations

import numpy as np


def svd_residual(cores, m_dims, n_dims, W: np.ndarray,
                 residual_rank: int) -> dict:
    """Closed-form residual compensation for one compressed layer.

    Returns the storage dict:
        cores   — the TT cores (unchanged; reuse existing certificates)
        u       — (M, r') residual left factor (float32)
        v       — (r', N) residual right factor (float32)
        s_in    — (N,) per-input-channel scale (currently all-ones; kept for
                  the activation-aware extension and for the certificate)
        s_out   — (M,) per-output-channel scale
    """
    from .kernels import get_backend
    k = get_backend()
    What = k.contract_cores(cores, m_dims, n_dims)
    R = W - What
    # Best rank-r' approximation of R: truncated SVD (Eckart-Young).
    u, s, vt = np.linalg.svd(R, full_matrices=False)
    r = int(min(residual_rank, len(s)))
    u = u[:, :r] * s[:r]          # fold singular values into U (storage: M*r)
    v = vt[:r, :]                 # storage: r*N
    return {
        "cores": cores,
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "s_in": np.ones(int(np.prod(n_dims)), dtype=np.float32),
        "s_out": np.ones(int(np.prod(m_dims)), dtype=np.float32),
        "residual_rank": r,
        "residual_energy": float(np.sum(s[:r] ** 2) / max(np.sum(s ** 2), 1e-30)),
    }


def apply(comp: dict, x: np.ndarray, m_dims, n_dims) -> np.ndarray:
    """Apply the compensated layer: y = TT(x) + U (V (S_in x)).

    `comp` is the dict returned by `svd_residual`.  Keeps the TT matvec
    structured (the certified path) and adds the low-rank residual branch.
    """
    from .kernels import get_backend
    k = get_backend()
    y_tt = k.tt_matvec(comp["cores"], x * comp["s_in"], m_dims, n_dims)
    y_res = comp["u"] @ (comp["v"] @ (comp["s_in"] * x))
    return y_tt * comp["s_out"] + y_res * comp["s_out"]


def to_dense(comp: dict, m_dims, n_dims) -> np.ndarray:
    """Materialize the compensated weight (for state-dict swap evals).

    W_comp = S_out (What + U V^T) S_in — what the dense fallback runs.
    """
    from .kernels import get_backend
    k = get_backend()
    What = k.contract_cores(comp["cores"], m_dims, n_dims)
    return comp["s_out"][:, None] * (What + comp["u"] @ comp["v"]) \
        * comp["s_in"][None, :]


def lipschitz_bound(comp: dict, m_dims, n_dims) -> float:
    """SOUND upper bound on the compensated layer's operator norm.

    ||S_out (TT + U V^T) S_in|| <= ||TT|| * max|s_out| * max|s_in|
                                 + ||U|| * ||V|| * max|s_out| * max|s_in|
    where ||TT|| <= lipschitz_product(cores) (the existing Layer-1 bound).
    """
    from .kernels import get_backend
    k = get_backend()
    l_tt = float(k.lipschitz_product(comp["cores"]))
    s_out = float(np.max(np.abs(comp["s_out"])))
    s_in = float(np.max(np.abs(comp["s_in"])))
    n_u = float(np.linalg.norm(comp["u"], 2))
    n_v = float(np.linalg.norm(comp["v"], 2))
    return s_out * s_in * (l_tt + n_u * n_v)


def compressed_params(comp: dict) -> int:
    """Honest storage count: TT cores + residual factors + scales."""
    cores = comp["cores"]
    tt = int(sum(int(np.prod(g.shape)) for g in cores))
    u = int(comp["u"].size)
    v = int(comp["v"].size)
    s = int(comp["s_in"].size + comp["s_out"].size)
    return tt + u + v + s


def layer_report(comp: dict, W: np.ndarray, m_dims, n_dims) -> dict:
    """One-glance per-layer record for the bench table and certificates."""
    from .kernels import get_backend
    k = get_backend()
    Wc = to_dense(comp, m_dims, n_dims)
    rel = float(np.linalg.norm(Wc - W) / max(np.linalg.norm(W), 1e-30))
    L = lipschitz_bound(comp, m_dims, n_dims)
    tight = float(k.operator_norm_tight(comp["cores"], m_dims, n_dims))
    # Tight norm of the compensated operator is upper-bounded by construction;
    # record the TT-part tight norm + the analytic bound for the table.
    return {
        "recon_rel_err": rel,
        "lipschitz_bound": L,
        "tt_tight_norm": tight,
        "residual_rank": comp["residual_rank"],
        "residual_energy": comp["residual_energy"],
        "params": compressed_params(comp),
        "dense_params": int(W.size),
        "sound": bool(np.isfinite(L)),
    }
