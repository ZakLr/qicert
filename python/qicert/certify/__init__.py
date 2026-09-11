"""Trust-augmentation modules (B14 family).

N3 (core certificates) stays in qicert/certify.py. This package adds:

- verify.py      — witness verifier + property tests (B14a)
- cross_check.py — interval/exact-arithmetic cross-check of certificate bounds (B14c)
- robustness.py  — statistical STL-robustness bound on the frozen eval split (B14b)
- manifest.py    — signed manifest of model/cert/split/config artifacts (B14h)
- guard.py       — runtime assertion guard (action in certified set per step) (B14h)

Lean prototype (B14d) lives at qicert/Cert.lean.
"""
from __future__ import annotations

__all__ = [
    "verify",
    "cross_check",
    "robustness",
    "manifest",
    "guard",
]
