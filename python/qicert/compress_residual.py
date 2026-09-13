"""Residual compensation for TT-compressed linear layers (N2'' repair arm).

Why this exists (2026-09-12, post go/no-go): uniform-ratio TT-SVD truncation
at 2x/3x compression measured eval acc 0.000 (down from the 0.447 fine-tuned
reference) — the uniform allocator removes signal, not redundancy, on this
backbone's flat spectra.  The pre-registered repair is closed-form residual
compensation (low-rank correction of the truncation residual, in the style
of recent quantized-plus-low-rank literature):

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
                 residual_rank: int,
                 activation_weight: np.ndarray | None = None,
                 scale: str = "absmax") -> dict:
    """Closed-form residual compensation for one compressed layer.

    activation_weight (N2R-v2, 2026-09-12): per-input-channel weight vector
    w (shape (N,), w >= 0).  The compensated operator remains a PLAIN
    matrix W_comp = What + C with C = U V (rank r'); the weighting changes
    WHICH C is chosen: C minimizes the activation-weighted error
        ||(W - What - C) diag(sqrt(w))||_F
    (GPTQ-style: spend the correction budget where the model actually
    computes).  Closed form: SVD of (W - What) diag(sqrt(w)) gives
    U S Vt; then C = U_r S_r Vt_r diag(1/sqrt(w)) with a zero-guard on
    channels whose weight is ~0 (their correction columns are set to 0:
    zero weight = "this channel is never exercised", so the optimum there
    is arbitrary and 0 keeps C finite).

    `scale` is a diagnostic label recorded in the dict ("absmax" | "mean"
    | "none") describing which calibration statistic built w; it does not
    change the operator.  Channel scales s_in/s_out stay identity, so the
    Layer-1 certificate of the compensated layer is exactly
    ||TT|| + ||U|| ||V|| (no extra diagonal factors).

    Returns the storage dict:
        cores   — the TT cores (unchanged; reuse existing certificates)
        u       — (M, r') residual left factor (float32, singular values folded)
        v       — (r', N) residual right factor (float32, de-weighted)
        s_in    — (N,) per-input-channel scale (identity; kept for interface
                  compatibility and future scale-aware arms)
        s_out   — (M,) per-output-channel scale (identity)
    """
    from .kernels import get_backend
    k = get_backend()
    What = k.contract_cores(cores, m_dims, n_dims)
    R = W - What
    if activation_weight is not None:
        w = np.asarray(activation_weight, dtype=np.float64).reshape(-1)
        if w.shape[0] != R.shape[1]:
            raise ValueError(
                f"activation_weight has {w.shape[0]} channels, "
                f"layer expects {R.shape[1]}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("activation_weight must be finite and >= 0")
        sw = np.sqrt(w)
        Rfit = R * sw[None, :]          # fit in the sqrt(w)-weighted space
    else:
        sw = None
        Rfit = R
    # Best rank-r' approximation of Rfit: truncated SVD (Eckart-Young in the
    # (activation-)weighted space).
    u, s, vt = np.linalg.svd(Rfit, full_matrices=False)
    r = int(min(residual_rank, len(s)))
    u = u[:, :r] * s[:r]          # fold singular values into U (storage: M*r)
    vt = vt[:r, :]
    if sw is not None:
        # De-weight V back into the unweighted (operator) space:
        # C = U Vt diag(1/sw); zero-weight channels get zero columns.
        floor = float(sw.max()) * 1e-8 if sw.size and sw.max() > 0 else 1.0
        inv_sw = np.where(sw > floor, 1.0 / np.maximum(sw, floor), 0.0)
        v = vt * inv_sw[None, :]
    else:
        v = vt
    return {
        "cores": cores,
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "s_in": np.ones(int(np.prod(n_dims)), dtype=np.float32),
        "s_out": np.ones(int(np.prod(m_dims)), dtype=np.float32),
        "residual_rank": r,
        "residual_energy": float(np.sum(s[:r] ** 2) / max(np.sum(s ** 2), 1e-30)),
        "activation_weighted": bool(sw is not None),
        "scale_stat": (scale if sw is not None else "none"),
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

    W_comp = What + U V — exactly what `apply` computes (verified by
    test_weighted_apply_matches_to_dense), so the dense fallback runs
    the same operator the certificate bounds.
    """
    from .kernels import get_backend
    k = get_backend()
    What = k.contract_cores(comp["cores"], m_dims, n_dims)
    return comp["s_out"][:, None] * (What + comp["u"] @ comp["v"]) \
        * comp["s_in"][None, :]


def lipschitz_bound(comp: dict, m_dims, n_dims) -> float:
    """SOUND upper bound on the compensated layer's operator norm.

    ||What + U V|| <= ||TT|| + ||U|| ||V||  (triangle ineq.; submultiplicativity)
    with ||TT|| <= lipschitz_product(cores) (the existing Layer-1 bound) and
    ||S_out|| = max|s_out|, ||S_in|| = max|s_in| multiplying exactly (diagonal
    operators).  The weighted arm keeps s_in/s_out at identity, so its bound
    is exactly ||TT|| + ||U|| ||V||.
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
