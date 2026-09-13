"""qicert — certified quantum-inspired compression for VLAMs.

Compress into QTT tensor networks, compile cross-modal interactions into
commuting-Pauli families, certify with a three-layer certificate stack, and
guard deployment with a syndrome-shadow monitor.

The four Python reference kernels are live in ``qicert.kernels``
(TT-cross + maxvol, commuting-Pauli compiler, Bayesian IQAE,
syndrome-shadow monitor). They are the conformance spec: any future
C++/CUDA-Q or Julia backend must reproduce the parity fingerprints in
``tests/test_kernels.py``.
"""

__version__ = "0.1.0.dev0"
