"""qicert.bench.all   reproduce every report table from a clean env.

Usage:
    python -m qicert.bench.all                 # every table (default)
    python -m qicert.bench.all --rows=smoke    # CI smoke set (tiny, fast)
    python -m qicert.bench.all --rows=compression-pareto   # one table
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
    args = ap.parse_args(argv)

    modules = [args.module] if args.module else MODULES
    out: list[str] = ["# qicert bench suite", ""]
    for name in modules:
        mod = importlib.import_module(f".{name}", package=__package__ or __name__)
        out.append(f"## bench.{name}")
        mod.run(args.rows, out)
        out.append("")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
