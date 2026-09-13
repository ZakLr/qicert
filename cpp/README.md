# qicert-cpp — Phase-2 native contraction kernel (SKELETON)

**Status: skeleton only. Not part of any Phase-1 measured path.**

Every number in the Phase-1 report was produced by the pure-Python pipeline
(`qicert/python/`, `bench/`). This directory exists to state, in code rather
than prose, where the Phase-2 latency lever lives. It compiles, it runs, and
it does exactly one thing — nothing more is claimed.

## Why a port is the Phase-2 lever (and only Phase-2)

Measured on the CPU edge profile (see `results/E6-latency/`, report
Fig. 7): TT per-layer contraction costs 3.1–15.2× the dense fp16 kernel
time at these shapes. The challenge's 100 ms per-step budget passes on every
measured path **in Python** — the margin comes from the profile, not from TT
being fast. The honest sentence in the report is: *"a C++ contraction kernel
is the Phase-2 latency lever."* This skeleton is that sentence, made real.

## What the port targets (Phase 2, with measured acceptance gates)

| Target | Python baseline (measured) | Phase-2 C++ gate |
|---|---|---|
| TT matvec per layer (bond 16) | 3.1–15.2× dense fp16 | ≤ 1.5× dense fp16 |
| Dense-cert re-derivation (SVD/layer, N9 loop) | ~minutes, NumPy | ≤ 30 s whole backbone |
| Guard step (ball check + manifest verify) | <1 ms | <100 µs |

No acceptance number counts until it is produced by `bench/latency_microbench.py`
against the same fixture as the Python rows — same hardware, same profile,
same ledger. Anything faster is marketing, not measurement.

## Layout

- `include/qicert/tt_matvec.hpp` — the core primitive: y += TT(W) @ x,
  cores in row-major, bond-ordered loops, no allocation in the hot loop.
- `src/tt_matvec.cpp` — the same, with the TODO markers where the Phase-2
  work goes (SIMD, open-loop ordering, fused dense-cert SVD hook).
- `CMakeLists.txt` — builds the static lib + a self-test binary.

## Build

CMake project (for Phase 2), but it also builds with one plain g++ line —
that is how it is verified on the dev machine (no cmake installed):

```bash
g++ -std=c++17 -O2 -Icpp/include cpp/src/tt_matvec.cpp cpp/src/selftest_main.cpp \
    -o cpp/qicert_cpp_selftest
./cpp/qicert_cpp_selftest
# -> qicert-cpp SKELETON (Phase-2 lever; no performance claims)
# -> selftest: identity-core property OK
```

With cmake (any machine that has it):

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build
./cpp/build/qicert_cpp_selftest   # same banner + property check
```

The self-test asserts the one property the skeleton must already honor
(the TT matvec on identity cores returns the input) so that "the skeleton
is structurally correct" is a testable statement, not an assertion. It makes
no performance claim and prints none.
