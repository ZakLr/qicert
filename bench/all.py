"""qicert.bench.all   reproduce every report table from a clean env.

Usage:
    python -m qicert.bench.all                     # every table (default)
    python -m qicert.bench.all --rows=smoke        # CI smoke set (tiny, fast)
    python -m qicert.bench.all --rows=compression-pareto   # one table
    python -m qicert.bench.all --backend=cpp       # C++/CUDA-Q kernels (Q19)
    python -m qicert.bench.all --sos-backend=julia # Layer-2a SOS (Q22 bridge)

Capture (Decision 2026-08-16 — record EVERYTHING, never re-run for data):
    python -m qicert.bench.all --rows=smoke --out results \
        --exp-id N2prime --seed 0 --run-tag smoke-test
    # writes results/<EXP_ID>/<RUN_ID>/run.json + config.json + system.json
    # + env.json + metrics.jsonl, and appends results/ledger.csv
"""
from __future__ import annotations

import argparse
import importlib
import sys

# Guard: never crash on a non-UTF-8 console (e.g. Windows cp1252). The
# bench output is ASCII-safe by contract, but this keeps any future
# non-ASCII cell from breaking the clean-env reproducibility gate.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

MODULES = [
    "compress",          # N2, N2', N3
    "certify_layers12",  # N4, N5
    "safety",            # N6
    "compiler",          # N7
    "monitor",           # N10
    "training",          # N11, N12
    "cross_track",       # N13
    "latency",           # <=100 ms profile
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qicert.bench.all")
    ap.add_argument("--rows", default="all",
                    help="'all', 'smoke', or a specific table name")
    ap.add_argument("--module", default=None, help="run a single bench module")
    ap.add_argument("--backend", default=None, choices=["python", "cpp"],
                    help="kernel backend; defaults to the active registry backend")
    ap.add_argument("--sos-backend", default=None, choices=["python", "julia"],
                    help="Layer-2a SOS solver backend")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="results root for run capture (writes run artifacts + ledger)")
    ap.add_argument("--exp-id", default=None,
                    help="experiment id recorded in run artifacts (e.g. N1, N2prime)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed recorded in run artifacts (default 0)")
    ap.add_argument("--run-tag", default="",
                    help="human tag appended to the run directory")
    ap.add_argument("--capture", default="default",
                    choices=["default", "extra", "heavy"],
                    help="capture depth (see docs/run-data-register.md)")
    args = ap.parse_args(argv)

    # Apply the selected backends before any bench module imports kernels.
    try:
        if args.backend:
            from qicert.kernels import set_active_backend
            set_active_backend(args.backend)
        if args.sos_backend:
            from qicert.kernels import set_active_sos_backend
            set_active_sos_backend(args.sos_backend)
    except ImportError as exc:  # pragma: no cover - package not installed
        print(f"error: qicert.kernels not importable: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: cannot select backend: {exc}", file=sys.stderr)
        return 2

    modules = [args.module] if args.module else MODULES
    from qicert.bench._base import BenchContext
    ctx = BenchContext(out=args.out, exp_id=args.exp_id, seed=args.seed,
                       run_tag=args.run_tag, capture=args.capture)
    out: list[str] = []
    if args.backend or args.sos_backend:
        out.append(f"# qicert bench suite (backend={args.backend or 'active'}, "
                   f"sos_backend={args.sos_backend or 'active'})")
    else:
        out.append("# qicert bench suite")
    if ctx.active:
        out.append(f"# capture: --out {ctx.out} (records run artifacts + ledger)")
    out.append("")
    for name in modules:
        mod = importlib.import_module(f".{name}", package=__package__ or __name__)
        out.append(f"## bench.{name}")
        mod.run(args.rows, out, ctx)
        out.append("")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
