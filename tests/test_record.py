"""Tests for qicert.record — the run-recorder layer (Decision 2026-08-16).

Contract: every benchmark writes run.json + config.json + system.json +
env.json + metrics.jsonl + a ledger.csv row, so no run ever needs re-running
for missing data. These tests pin the artifact shape and the guarantees.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from qicert.record import (
    RunRecorder,
    config_hash,
    env_info,
    gpu_info,
    git_info,
    system_info,
)


@pytest.fixture()
def out_root(tmp_path):
    return str(tmp_path / "results")


def test_recorder_writes_all_artifacts(out_root):
    rec = RunRecorder(exp_id="N1", seed=3,
                      config={"model": "minivla", "lr": 1e-4},
                      out_root=out_root, run_tag="sliceA")
    assert rec.run_dir.exists()
    assert (rec.run_dir / "run.json").exists()
    assert (rec.run_dir / "config.json").exists()
    assert (rec.run_dir / "system.json").exists()
    assert (rec.run_dir / "env.json").exists()
    assert (rec.run_dir / "metrics.jsonl").exists()

    run = rec.summary()
    assert run["experiment_id"] == "N1"
    assert run["seed"] == 3
    assert run["status"] == "running"
    assert run["config_hash"] == config_hash({"model": "minivla", "lr": 1e-4})

    cfg = json.loads((rec.run_dir / "config.json").read_text())
    assert cfg["config"]["lr"] == 1e-4

    sys_info = json.loads((rec.run_dir / "system.json").read_text())["system"]
    assert "platform" in sys_info and "gpu" in sys_info
    env = json.loads((rec.run_dir / "env.json").read_text())["env"]
    assert "numpy" in env

    rec.metric(step=0, loss=1.25)
    rec.metric(step=1, loss=np.float64(1.1), grad_norm=0.01)
    lines = (rec.run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3  # run_start + 2 metrics
    assert '"t_wall"' in lines[1]  # canonical_json sorts keys; every line has t_wall
    assert json.loads(lines[2])["loss"] == 1.1

    rec.finalize(status="completed", results={"success": 0.9},
                 kill_criterion="R1", kill_verdict="not triggered")
    run2 = rec.summary()
    assert run2["status"] == "completed"
    assert run2["results"]["success"] == 0.9
    assert run2["kill_criterion"] == "R1"

    # ledger row exists and is keyed on run_id
    ledger_path = os.path.join(out_root, "ledger.csv")
    assert os.path.exists(ledger_path)
    with open(ledger_path) as fh:
        rows = [r for r in __import__("csv").reader(fh)]
    assert rows[0][0] == "experiment_id"
    assert rows[1][0] == "N1" and rows[1][1] == rec.run_id
    assert float(rows[1][6]) == pytest.approx(run2["wall_sec"], abs=1.0)


def test_finalize_idempotent(out_root):
    rec = RunRecorder("N2", 0, {"a": 1}, out_root=out_root)
    rec.finalize(status="completed", results={"x": 1})
    rec.finalize(status="completed", results={"x": 1})  # second call: no dup
    with open(os.path.join(out_root, "ledger.csv")) as fh:
        rows = list(__import__("csv").reader(fh))
    assert len(rows) == 2  # header + 1 row


def test_config_hash_deterministic_and_key_order_insensitive():
    a = config_hash({"b": 2, "a": [1, 2], "c": {"d": 4}})
    b = config_hash({"c": {"d": 4}, "a": [1, 2], "b": 2})
    assert a == b
    assert config_hash({"x": np.float32(0.5)}) == config_hash({"x": 0.5})
    assert len(a) == 16


def test_run_tag_sanitized(out_root):
    rec = RunRecorder("N1", 0, {}, out_root=out_root, run_tag="my run/tag::x")
    assert rec.run_dir.name.startswith(rec.run_id)
    assert "/" not in rec.run_dir.name.split(rec.run_id, 1)[1]


def test_system_and_env_capture_no_gpu_required():
    info = system_info()
    assert isinstance(info, dict)
    assert "python" in info and "gpu" in info
    assert isinstance(info["gpu"], dict)  # may be {} when no GPU
    env = env_info()
    assert env["numpy"] is not None
    assert env["torch"] is None or isinstance(env["torch"], str)


def test_git_info():
    info = git_info()
    assert "git_commit" in info and "git_dirty" in info
    # when run inside the qicert checkout, commit is present
    assert info["git_commit"] is None or len(info["git_commit"]) >= 7


def test_recorder_without_torch_or_nvidia_smi(out_root, monkeypatch):
    """The recorder must work in environments with no GPU tooling."""
    monkeypatch.setenv("PYTHONPATH", "")
    gpu = gpu_info()  # must not raise
    assert isinstance(gpu, dict)
    rec = RunRecorder("kernel-smoke", 0, {"check": "no-gpu"}, out_root=out_root)
    rec.metric(check="tt_svd", value=1e-10)
    rec.finalize(status="completed")
    assert rec.summary()["status"] == "completed"


def test_sample_power_does_not_raise(out_root):
    rec = RunRecorder("N1", 0, {}, out_root=out_root)
    rec.sample_power()  # may record None power when no GPU tooling
    rec.sample_power()
    rec.finalize()
    lines = (rec.run_dir / "metrics.jsonl").read_text().splitlines()
    assert any('"gpu_power_w"' in ln for ln in lines)
