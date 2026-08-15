"""qicert — certified quantum-inspired compression for VLAMs.

Compress into QTT tensor networks, compile cross-modal interactions into
commuting-Pauli families, certify with a three-layer certificate stack, and
guard deployment with a syndrome-shadow monitor.

Phase 0.5 (2026-08-11): the four Python reference kernels are live in
``qicert.kernels`` (TT-cross + maxvol, commuting-Pauli compiler, Bayesian
IQAE, syndrome-shadow monitor). They are the conformance spec: the C++/CUDA-Q
backend (Q19) and Julia SOS bridge (Q22) must reproduce the parity
fingerprints in ``tests/test_kernels.py``.
"""

__version__ = "0.1.0.dev0"
