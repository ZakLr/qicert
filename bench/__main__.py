"""python -m qicert.bench   same as qicert.bench.all."""
import sys

try:
    from qicert.bench.all import main
except ImportError:
    from bench.all import main

if __name__ == "__main__":
    sys.exit(main())
