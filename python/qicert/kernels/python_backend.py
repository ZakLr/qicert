"""Pure-Python (numpy/scipy) reference implementation of the qicert kernels.

Decision (2026-08-11): this is the conformance spec. The C++/CUDA-Q backend
(Q19) and Julia SOS bridge (Q22) must reproduce these results within the
tolerances in ``tests/test_kernels.py``.

Implemented:
  * Pillar A  — tt_svd, tt_cross (query-based DMRG-cross, maxvol),
                contract_cores, tt_matvec, lipschitz_product,
                operator_norm_tight (Rayleigh-quotient power iteration)
  * Pillar B  — pauli_grouping (greedy commutation), pauli_matrix,
                pauli_diagonalize (joint eigenbasis == the Clifford frame)
  * Pillar C  — iqae (Bayesian posterior over the amplitude angle)
  * Monitor   — shadow_statistics (median-of-means), shadow_syndrome

All kernels are numpy/scipy only (CI and bench smoke never import torch).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy import stats

from .base import (
    Cores,
    Diagonalization,
    IQAEInterval,
    KernelError,
    KernelSet,
    PauliGrouping,
    ShadowVerdict,
)
from . import register_backend


def _product(xs) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _check_dims(matrix, m_dims, n_dims) -> None:
    M, N = matrix.shape
    if _product(m_dims) != M or _product(n_dims) != N:
        raise ValueError(
            f"mode dims {m_dims}x{n_dims} do not match matrix shape {matrix.shape}"
        )


class PythonKernelSet(KernelSet):
    """The reference kernel implementation (numpy/scipy)."""

    name = "python"

    # ------------------------------------------------------------------
    # Pillar A — compression + Layer-1 certificates
    # ------------------------------------------------------------------

    def tt_svd(self, matrix, m_dims, n_dims, ranks):
        """Sequential TT-SVD of a matrix into TT-matrix cores.

        Reference decomposition; exact reconstruction for exact-TT input.
        Cores are 4-way arrays ``(r_{k-1}, m_k, n_k, r_k)`` with r_0 = r_d = 1.
        """
        _check_dims(matrix, m_dims, n_dims)
        d = len(m_dims)
        if len(n_dims) != d:
            raise ValueError("m_dims and n_dims must have equal length")
        shape = [int(m_dims[k]) * int(n_dims[k]) for k in range(d)]
        # internal bonds: exactly d-1 ranks (pad/truncate caller-provided tuple)
        rr = [int(x) for x in ranks][:d - 1]
        while len(rr) < d - 1:
            rr.append(rr[-1] if rr else 1)
        r = [1] + [min(int(x), shape[k]) for k, x in enumerate(rr)] + [1]

        # W -> tensor (m_1..m_d, n_1..n_d) -> interleave -> fused modes (m_k n_k)
        Wt = matrix.reshape(tuple(m_dims) + tuple(n_dims))
        order = []
        for k in range(d):
            order += [k, d + k]
        fused = Wt.transpose(order).reshape(shape)

        cores: list[np.ndarray] = []
        # Sequential TT-SVD: at each step the *fused* left modes (r_{k-1} s_k)
        # must be a flat 2-D matrix for SVD — `fused` is d-dimensional for
        # deeper splits, so reshape explicitly at every step.
        cur = fused.reshape((shape[0], -1))
        achieved = [1]  # r_0 = 1; tracks ranks actually used (SVD can cap)
        for k in range(d - 1):
            # rk is capped by BOTH current dimensions: when the requested rank
            # exceeds cur.shape[1] (e.g. small GQA k/v projections at deep
            # splits), U[:, :rk] silently returns fewer columns and the
            # reshape below would mismatch. Achieved ranks propagate to the
            # next core's left bond and to the final core.
            rk = min(r[k + 1], cur.shape[0], cur.shape[1])
            rk = max(int(rk), 1)
            U, S, Vt = linalg.svd(cur, full_matrices=False)
            U, S, Vt = U[:, :rk], S[:rk], Vt[:rk, :]
            # U rows are (achieved[-1] * s_k) with r outer, s inner, and
            # s_k = m_k n_k with m outer / n inner (from the interleave), so
            # the reshape to the 4-way core is exact.
            cores.append(U.reshape((achieved[-1], m_dims[k], n_dims[k], rk)))
            achieved.append(rk)
            cur = (S[:, None] * Vt).reshape((rk * shape[k + 1], -1))
        cores.append(cur.reshape((achieved[-1], m_dims[d - 1],
                                  n_dims[d - 1], 1)))
        return Cores(cores, tuple(int(x) for x in m_dims),
                     tuple(int(x) for x in rr), source=self.name)

    # -- TT-cross ---------------------------------------------------------

    def tt_cross(self, matrix, m_dims, n_dims, ranks):
        """Query-based DMRG-cross (Oseledets--Tyrtyshnikov style).

        Works on the fused tensor with mode sizes ``s_k = m_k * n_k`` and an
        entry oracle into ``matrix`` — the dense matrix is never formed. The
        alternating sweeps refine left/right skeleton index sets via maxvol;
        the final pass builds interpolated cores with the recursive-cross
        normalization ``C_k = F_k @ (F_k[I_k])^{-1}`` so the product
        telescopes exactly on exact-TT input.
        """
        _check_dims(matrix, m_dims, n_dims)
        d = len(m_dims)
        if len(n_dims) != d:
            raise ValueError("m_dims and n_dims must have equal length")
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
            return float(matrix[row, col])

        fused_cores = self._cross_als(f, shape, ranks)
        out = []
        for k, g in enumerate(fused_cores):
            rk_prev, sk, rk = g.shape
            out.append(np.ascontiguousarray(
                g.reshape(rk_prev, int(m_dims[k]), int(n_dims[k]), rk)))
        rr = [int(x) for x in ranks][:d - 1]
        while len(rr) < d - 1:
            rr.append(rr[-1] if rr else 1)
        return Cores(out, tuple(int(x) for x in m_dims),
                     tuple(int(x) for x in rr), source=self.name)

    def _cross_als(self, f, shape, ranks, sweeps: int = 3,
                   tol: float = 1.05, seed: int = 0):
        """DMRG-cross over the fused tensor with ranks r_0 = r_d = 1.

        Skeletons: L[k] = r_k index tuples over modes 0..k-1;
                   R[k] = r_{k+1} tuples over modes k+1..d-1.
        """
        d = len(shape)
        rr = [int(x) for x in ranks][:d - 1]
        while len(rr) < d - 1:
            rr.append(rr[-1] if rr else 1)
        r = [1] + [min(int(x), int(shape[k])) for k, x in enumerate(rr)] + [1]
        rng = np.random.default_rng(seed)

        def rand_multi(modes, count):
            if not modes:
                return [()]
            return [tuple(int(x) for x in rng.integers(0, [shape[m] for m in modes]))
                    for _ in range(count)]

        def fused_of(multi) -> int:
            idx = 0
            for m, s in zip(multi, shape):
                idx = idx * int(s) + int(m)
            return idx

        def frame_fwd(k, L, R):
            """Forward (row-major) frame: rows (a, i), cols b."""
            la, rb = r[k], r[k + 1]
            F = np.empty((la * shape[k], rb), dtype=float)
            for a in range(la):
                base = L[k][a]
                for i in range(shape[k]):
                    row = a * shape[k] + i
                    for b in range(rb):
                        F[row, b] = f(fused_of(base + (i,) + R[k][b]))
            return F

        def frame_bwd(k, L, R):
            """Backward (transposed) frame: rows (b, i), cols a."""
            la, rb = r[k], r[k + 1]
            F = np.empty((rb * shape[k], la), dtype=float)
            for b in range(rb):
                base = R[k][b]
                for i in range(shape[k]):
                    row = b * shape[k] + i
                    for a in range(la):
                        F[row, a] = f(fused_of(L[k][a] + (i,) + base))
            return F

        # initialize skeletons (L[0] and R[d-1] are the single empty tuple)
        L = [rand_multi(list(range(k)), r[k]) for k in range(d)]
        R = [rand_multi(list(range(k + 1, d)), r[k + 1]) for k in range(d)]

        for _sweep in range(sweeps):
            # ---- forward pass: refine L given R ----
            for k in range(d - 1):
                F = frame_fwd(k, L, R)
                rows = self._maxvol(F, tol=tol)
                pairs = [divmod(int(row), shape[k]) for row in rows]
                L[k + 1] = [L[k][a] + (i,) for a, i in pairs]
            # ---- backward pass: refine R given L ----
            for k in range(d - 1, 0, -1):
                F = frame_bwd(k, L, R)
                rows = self._maxvol(F, tol=tol)
                pairs = [divmod(int(row), shape[k]) for row in rows]
                R[k - 1] = [(i,) + R[k][b] for b, i in pairs]

        # ---- final pass: interpolated cores with recursive-cross norm ----
        # Skeletons update in lockstep with the cores so the product
        # telescopes exactly on exact-TT input (C_k = F_k (F_k[I_k])^{-1}).
        cores = [None] * d
        for k in range(d):
            F = frame_fwd(k, L, R)
            if k < d - 1:
                rows = self._maxvol(F, tol=tol)
                W = F[rows]                      # r_{k+1} x r_{k+1} pivot block
                C = linalg.solve(W.T, F.T).T     # F @ W^{-1}
                pairs = [divmod(int(row), shape[k]) for row in rows]
                L[k + 1] = [L[k][a] + (i,) for a, i in pairs]
            else:
                C = F                            # last core stays raw
            cores[k] = C.reshape((r[k], shape[k], r[k + 1]))
        return cores

    def _maxvol(self, A, tol: float = 1.05) -> np.ndarray:
        """Maximal-volume row selection for a tall matrix A (n x p), n >= p."""
        A = np.asarray(A, dtype=float)
        n, p = A.shape
        if p > n:
            raise KernelError(f"maxvol needs n>=p, got {A.shape}")
        _, _, piv = linalg.qr(A.T, pivoting=True, mode="economic")
        rows = list(piv[:p])
        B = A[rows]
        for _ in range(80):
            C = linalg.solve(B.T, A.T).T              # A @ B^{-1}, (n, p)
            mag = np.abs(C)
            k = int(np.argmax(mag))
            if mag.flat[k] <= 1.0 + tol:
                break
            i_star, j_star = np.unravel_index(k, mag.shape)
            if i_star in rows:
                break
            rows[j_star] = i_star
            B = A[rows]
        return np.array(rows, dtype=int)

    # -- evaluation + norms -------------------------------------------------

    def contract_cores(self, cores, m_dims, n_dims) -> np.ndarray:
        """Contract cores back to the dense weight matrix (M x N)."""
        d = len(cores)
        t = cores[0]
        for k in range(1, d):
            t = np.tensordot(t, cores[k], axes=([-1], [0]))
        # t axes are (r_0, m_1, n_1, m_2, n_2, ..., m_d, n_d, r_d) with
        # r_0 = r_d = 1; move the n-axes to the end so reshape gives the
        # row-major matrix convention.
        m_axes = [1 + 2 * k for k in range(d)]
        n_axes = [2 + 2 * k for k in range(d)]
        t = t.transpose((0,) + tuple(m_axes) + tuple(n_axes) + (2 * d + 1,))
        return t.reshape((_product(m_dims), _product(n_dims)))

    def tt_matvec(self, cores, v, m_dims, n_dims) -> np.ndarray:
        """TT-matrix x vector without forming the dense matrix.

        Runs right-to-left; cores[k] = (r_k, m_{k+1}, n_{k+1}, r_{k+1}).
        Running tensor invariant before step k:
            t = (r_{k+1}, n_1..n_{k+1}, m_{k+2}..m_d)
        """
        d = len(cores)
        t = v.reshape((1,) + tuple(n_dims))
        for k in range(d - 1, -1, -1):
            G = cores[k]                              # (r_k, m, n, r_{k+1})
            r_cur, mk, nk, r_next = G.shape
            # pair G axis 3 (r_{k+1}) with t axis 0; G axis 2 (n_{k+1}) with
            # t axis (1 + k); remaining t axes: n_1..n_k, m_{k+2}..m_d
            U = np.tensordot(G, t, axes=([3, 2], [0, 1 + k]))
            nb = t.shape[1:k + 1]                     # n_1..n_k
            mb = t.shape[k + 2:]                      # m_{k+2}..m_d
            U = U.reshape((r_cur, mk) + nb + mb)
            t = np.moveaxis(U, 1, 1 + len(nb))
        return t.reshape(_product(m_dims))

    def lipschitz_product(self, cores) -> float:
        """Layer-1 bound: L = prod_k ||G_k||_2 (core spectral norms)."""
        L = 1.0
        for g in cores:
            L *= self._core_norm(g)
        return float(L)

    def _core_norm(self, g: np.ndarray) -> float:
        if g.ndim == 2:
            return float(np.linalg.norm(g, 2))
        if g.ndim == 4:
            r_prev, mk, nk, rk = g.shape
            return float(np.linalg.norm(g.reshape(r_prev * mk, nk * rk), 2))
        if g.ndim == 3:
            r_prev, s, rk = g.shape
            return float(np.linalg.norm(g.reshape(r_prev * s, rk), 2))
        raise ValueError(f"unsupported core ndim {g.ndim}")

    def operator_norm_tight(self, cores, m_dims, n_dims, iters: int = 100) -> float:
        """Tight per-layer operator norm via Rayleigh-quotient power iteration.

        Converges at rate (sigma_2/sigma_1)^{2k}, so 100 iters is ample for
        N3's tightness ratio even with near-degenerate spectra.
        """
        d = len(cores)
        if d == 0:
            return 0.0
        N = _product(n_dims)
        rng = np.random.default_rng(0)
        v = rng.standard_normal(N)
        v /= np.linalg.norm(v)
        tcores = [np.swapaxes(g, 1, 2) if g.ndim == 4 else g.T for g in cores]
        lam = 0.0
        for _ in range(iters):
            w = self.tt_matvec(tcores, self.tt_matvec(cores, v, m_dims, n_dims),
                               n_dims, m_dims)
            lam = float(w @ v) / float(v @ v)         # Rayleigh quotient
            nrm = float(np.linalg.norm(w))
            if nrm < 1e-300:
                break
            v = w / nrm
        return float(np.sqrt(max(lam, 0.0)))

    # ------------------------------------------------------------------
    # Pillar B — commuting-Pauli compiler
    # ------------------------------------------------------------------

    _PAULI = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}

    def pauli_matrix(self, pauli: str, k: int) -> np.ndarray:
        """2^k x 2^k Hermitian matrix for a k-qubit Pauli string."""
        eye = np.eye(2)
        sx = np.array([[0.0, 1.0], [1.0, 0.0]])
        sz = np.array([[1.0, 0.0], [0.0, -1.0]])
        sy = 1j * sx @ sz
        out = np.array([[1.0]])
        for ch in pauli:
            out = np.kron(out, {"I": eye, "X": sx, "Y": sy, "Z": sz}[ch])
        return out

    def _commute(self, p: str, q: str) -> bool:
        acc = 0
        for a, b in zip(p, q):
            xa, za = self._PAULI[a]
            xb, zb = self._PAULI[b]
            acc += xa * zb + za * xb
        return acc % 2 == 0

    def pauli_grouping(self, table, k: int) -> PauliGrouping:
        """Greedy commutation grouping (polynomial, not optimal) + pruning."""
        families: list[list[tuple[float, str]]] = []
        for coef, pauli in table:
            placed = False
            for fam in families:
                if all(self._commute(pauli, q) for _, q in fam):
                    fam.append((float(coef), pauli))
                    placed = True
                    break
            if not placed:
                families.append([(float(coef), pauli)])
        pruning = [sum(abs(c) for c, _ in fam) for fam in families]
        return PauliGrouping(families, pruning, len(families))

    def pauli_diagonalize(self, family, k: int) -> Diagonalization:
        """Joint eigenbasis via successive eigenspace refinement.

        Because the family commutes, refining the basis within each current
        eigenspace never breaks previously diagonalized strings. The resulting
        unitary U is the concrete realization of the report's 'single Clifford
        frame' (Pauli-to-Z diagonalization) for small k.
        """
        dim = 2 ** k
        mats = [self.pauli_matrix(p, k) for _, p in family]
        U = np.eye(dim, dtype=complex)
        blocks: list[np.ndarray] = [np.arange(dim)]
        for M in mats:
            new_U = U.copy()
            new_blocks: list[np.ndarray] = []
            for block in blocks:
                sub = U[:, block].conj().T @ M @ U[:, block]
                sub = (sub + sub.conj().T) / 2
                evals, evecs = np.linalg.eigh(sub)
                for ev in np.unique(np.round(evals, 12)):
                    mask = np.isclose(evals, ev, atol=1e-9)
                    new_blocks.append(block[mask])
                new_U[:, block] = U[:, block] @ evecs
            U = new_U
            blocks = new_blocks
        diags = []
        for M in mats:
            d = np.real_if_close(np.diag(U.conj().T @ M @ U))
            diags.append(np.round(d, 12))
        return Diagonalization(U, diags)

    # ------------------------------------------------------------------
    # Pillar C — Bayesian IQAE
    # ------------------------------------------------------------------

    def iqae(self, measure, budget: int = 200, shots: int = 30,
             grid: int = 4096, seed: int = 0) -> IQAEInterval:
        """Simulated Bayesian IQAE (Grinko-style posterior over the angle).

        ``measure(depth, shots) -> int`` returns successes from ``shots``
        measurements at Grover depth ``depth``. Budget is spent in rounds of
        interleaved anchor (depth 0) and growing odd depth (0, 1, 0, 3, 0, 7,
        0, 15, ...) — the Grinko schedule. The depth-0 anchors prevent the
        phase-wrapping aliasing of the growing-depth likelihoods from locking
        the posterior onto a spurious sharp mode.
        """
        theta = np.linspace(0.0, np.pi / 2, grid)
        post = np.full(grid, 1.0 / grid)
        depths: list[int] = []
        remaining = int(budget)
        k = 1
        round_i = 0
        while remaining > 0:
            depth = 0 if round_i % 2 == 0 else k
            # aliasing guard: keep >= 8 grid points per likelihood oscillation
            # so the posterior can never vanish to zero on the whole grid
            if (2 * depth + 1) > max(16, grid // 8):
                break
            s = int(measure(depth, shots))
            depths.append(depth)
            p_succ = np.clip(np.sin((2 * depth + 1) * theta) ** 2, 1e-15, 1 - 1e-15)
            like = stats.binom.pmf(int(s), shots, p_succ)
            post = post * like
            total = post.sum()
            if total <= 0 or not np.isfinite(total):
                break
            post /= total
            remaining -= shots
            round_i += 1
            if round_i % 2 == 0:
                k = 2 * k + 1
        p_grid = np.sin(theta) ** 2
        map_idx = int(np.argmax(post))
        p_map = float(p_grid[map_idx])
        cdf = np.cumsum(post)
        cdf = cdf / cdf[-1]          # normalize to a proper CDF
        # interpolated quantiles (grid-snapping can miss the boundary by a
        # single grid step, which breaks coverage at 5 rounds)
        lo = float(np.interp(0.025, cdf, p_grid))
        hi = float(np.interp(0.975, cdf, p_grid))
        return IQAEInterval(p_map, lo, hi, len(depths) * shots,
                            max(depths) if depths else 0, theta, post)

    # ------------------------------------------------------------------
    # Monitor — syndrome-shadow runtime guard
    # ------------------------------------------------------------------

    def shadow_statistics(self, x, projections, baseline_mean, baseline_std,
                          blocks: int = 8, fpr: float = 0.01) -> ShadowVerdict:
        """Median-of-means over M random projections (Huang--Kueng--Preskill).

        ``baseline_mean`` / ``baseline_std`` are the reference statistics of a
        single projection under the in-distribution input (callers compute them
        on a baseline batch once). The gate alarms when the median-of-means
        deviates beyond the fpr-normal quantile in std errors.
        """
        s = np.asarray(projections, dtype=float) @ np.asarray(x, dtype=float)
        M = len(s)
        if M < 2 * blocks:
            blocks = max(1, M // 2)
        idx = np.array_split(np.arange(M), blocks)
        block_means = np.array([s[i].mean() for i in idx])
        med = float(np.median(block_means))
        se = float(baseline_std) / np.sqrt(len(block_means))
        z = (med - float(baseline_mean)) / max(se, 1e-12)
        thr = float(stats.norm.ppf(1 - fpr))
        return ShadowVerdict(z, thr, bool(z > thr))

    def shadow_syndrome(self, diag_patterns) -> int:
        """PDU hygiene: syndrome = hash of the compiled families' diagonal
        patterns; a change between inference steps gray-lists the packet."""
        h = hash(tuple(tuple(np.round(p, 6).tolist()) for p in diag_patterns))
        return h & ((1 << 31) - 1)


register_backend("python", PythonKernelSet)
