"""step0 module contract tests (no checkpoint needed — run() must not touch
the weights when the row is not selected, keeping CI/clean-env torch-free)."""
from __future__ import annotations

from bench import step0


def test_row_not_selected_does_not_touch_checkpoint(tmp_path):
    """--rows=smoke (or anything except layer-smoke) must return immediately."""
    out: list[str] = []
    # Would raise FileNotFoundError if it tried to load the checkpoint.
    step0.run("smoke", out, None)
    step0.run("kernel-smoke", out, None)
    step0.run("baseline-int8", out, None)
    assert out == []


def test_layer_key_shape_dims_consistent():
    """Smoke sanity: mode dims must factor the layer shape (896 = 32*28)."""
    assert step0.M_DIMS[0] * step0.M_DIMS[1] == 896
    assert step0.N_DIMS[0] * step0.N_DIMS[1] == 896
