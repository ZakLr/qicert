"""Phase-0 smoke tests: the bench-suite contract and package imports."""
import subprocess
import sys


def test_bench_smoke_runs():
    """qicert.bench.all --rows=smoke must complete and print tables."""
    r = subprocess.run(
        [sys.executable, "-m", "qicert.bench.all", "--rows=smoke"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "# qicert bench suite" in r.stdout
    assert "bench.compress" in r.stdout
    assert "pending" in r.stdout


def test_package_imports():
    import qicert
    from qicert.safety import AD_SPECS, ROBOTICS_SPECS
    assert qicert.__version__
    assert len(AD_SPECS) == 4 and len(ROBOTICS_SPECS) == 3


def test_lipschitz_constant_computes():
    import numpy as np
    from qicert.compress import lipschitz_constant
    cores = [np.eye(4) * 0.5 for _ in range(3)]
    assert abs(lipschitz_constant(cores) - 0.5 ** 3) < 1e-12
