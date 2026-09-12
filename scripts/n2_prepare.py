"""N2 launch preparation + self-contained run harness.

Two things in one file, deliberately separated so the "prepare" step is
CPU-only and the "run" step is the only thing that touches the GPU:

1. PREPARE  — compute the exact N2 run parameters from the checkpoint + the
   existing N1v2 handoff (seed0.pt, eval_batch_seed0.pt, baseline_seed0.json),
   emit a fully-determined launch blueprint (exp_id, seeds, backbones, plans,
   layer scope, bit-ordering, kernel, batches, capture, env overrides, expected
   point count), and write results/N2/blueprint.json + a per-seed handoff
   status file.  No GPU, no VLA load, no dataset.

2. RUN      — the function `run_n2_from_blueprint(blueprint, ctx)` is the
   exact function that the bench/all.py + n2_sweep module already implements;
   this file does NOT reimplement the VLA eval.  It only documents the
   self-contained invocation shape so that later, when we want to press Go,
   we can reproduce any N2 point from the blueprint without re-deriving the
   parameters.

Usage (CPU-only prepare, now):
    PYTHONPATH= python scripts/n2_prepare.py

Usage (later, when ready to run — one seed at a time, Docker, unbuffered):
    See the command block printed by the script at the end of its run.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (decision 2026-08-16 / 2026-09-11): checkpoint + N1v2 handoff dir
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "weights" / "ckpt" / "checkpoints" / "step-122500-epoch-55-loss=0.0743.pt"
HANDOFF = REPO / "results" / "N1v2-ckpt"
OUT = REPO / "results"
LOG_DIR = OUT / "logs"
N2_DIR = OUT / "N2"

# N2 sweep contract (from bench/n2_sweep.py, the single source of truth):
#   BOND_PLANS = (0.02, 0.04, 0.08, 0.16, 0.33, 0.50)   # param fractions
#   BACKBONES  = ("TT", "QTT")                              # d=2 vs d=4
#   kernel     = tt_svd (deterministic reference; tt_cross = Q19 CUDA-Q port)
#   ordering   = bit-reversed (N2prime winner, 178c849b6525)
#   eval       = SAME batch as N1 (matched budget), on FINE-TUNED weights
#   layers     = all LLM linear projections (q/k/v/o/gate/up/down, 24 layers)
#   per-point  = run.json + config.json + system.json + env.json + metrics.jsonl
#               + ledger.csv row, plus per-layer certificate metrics

BOND_PLANS = (0.02, 0.04, 0.08, 0.16, 0.33, 0.50)
BACKBONES = ("TT", "QTT")
ORDERING = "bit-reversed"          # N2prime winner
KERNEL = "tt_svd (deterministic reference; tt_cross = Q19 CUDA-Q port)"
LAYERS_SCOPE = "all LLM linear projections (24 layers x q/k/v/o/gate/up/down)"
POINTS_PER_SEED = len(BOND_PLANS) * len(BACKBONES)  # 12

# Seeds available for a CLEAN matched-budget handoff (need seedN.pt +
# eval_batch_seedN.pt + baseline_seedN.json in the N1v2 handoff dir).
def handoff_status(seed: int) -> dict:
    pt = HANDOFF / f"seed{seed}.pt"
    batch = HANDOFF / f"eval_batch_seed{seed}.pt"
    # the file is literally baseline_seed0.json / baseline_seed1.json / ...
    baseline = HANDOFF / f"baseline_seed{seed}.json"
    ft_acc = None
    if baseline.exists():
        try:
            ft_acc = float(json.loads(baseline.read_text())["eval_acc_finetuned"])
        except Exception:
            pass
    return {
        "seed": seed,
        "seed_pt_exists": pt.exists(),
        "seed_pt_mb": round(pt.stat().st_size / 1024**2, 1) if pt.exists() else None,
        "eval_batch_exists": batch.exists(),
        "eval_batch_mb": round(batch.stat().st_size / 1024**2, 1) if batch.exists() else None,
        "baseline_json_exists": baseline.exists(),
        "baseline_ft_acc": ft_acc,
        "clean_handoff": pt.exists() and batch.exists() and baseline.exists(),
    }


def pre_launch_snapshot() -> dict:
    """Resource + handoff snapshot recorded BEFORE the run, for provenance."""
    snap = {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": _torch_version(),
        "cuda_available": _cuda_available(),
    }
    if _cuda_available():
        import torch
        p = torch.cuda.get_device_properties(0)
        snap["gpu_name"] = torch.cuda.get_device_name(0)
        snap["gpu_total_memory_gb"] = round(p.total_memory / 1024**3, 3)
        snap["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024**2, 1)
        snap["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1024**2, 1)
        snap["gpu_free_estimate_mb"] = round((p.total_memory - torch.cuda.memory_allocated(0)) / 1024**2, 1)
    snap["docker_image_present"] = _docker_image_present()
    snap["docker_image_tag"] = _docker_image_tag()
    snap["prismatic_importable_on_host"] = _prismatic_importable_on_host()
    snap["n1v2_handoff_dir"] = str(HANDOFF)
    snap["n1v2_handoff_seed0"] = handoff_status(0)
    snap["n1v2_handoff_seed1"] = handoff_status(1)
    snap["n1v2_handoff_seed2"] = handoff_status(2)
    snap["ckpt_exists"] = CKPT.exists()
    snap["ckpt_size_mb"] = round(CKPT.stat().st_size / 1024**2, 1) if CKPT.exists() else None
    snap["n2prime_winner"] = ORDERING
    snap["n2prime_run_id"] = "178c849b6525"
    snap["n2prime_recon_err"] = {"bit-reversed": 0.9185, "interleaved": 0.9365, "natural": 0.9387}
    snap["n2_sweep_kernel"] = KERNEL
    snap["n2_sweep_backbones"] = list(BACKBONES)
    snap["n2_sweep_ordering"] = ORDERING
    snap["n2_sweep_bond_plans"] = list(BOND_PLANS)
    snap["n2_sweep_layers_scope"] = LAYERS_SCOPE
    snap["n2_sweep_points_per_seed"] = POINTS_PER_SEED
    snap["n2_sweep_eval_protocol"] = (
        "SAME eval batch as N1 (matched budget), on FINE-TUNED weights, "
        "per-point recorder artifacts + ledger row."
    )
    return snap


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:
        return "not-imported"


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _docker_image_present() -> bool:
    try:
        import subprocess
        out = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return any("qicert" in line or "cuda" in line.lower() for line in out.stdout.splitlines())
        return False
    except Exception:
        return False


def _docker_image_tag() -> str | None:
    try:
        import subprocess
        out = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.Size}}"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if "qicert" in line or "cuda" in line.lower():
                    parts = line.split()
                    return parts[0] if parts else None
        return None
    except Exception:
        return None


def _prismatic_importable_on_host() -> bool:
    """Prismatic importability check against the interpreter that actually runs
    the benches (.venv312), not whatever Python the script itself is under —
    the 2026-09-12 prep recorded `false` from system Python 3.14 while
    .venv312 (the interpreter that successfully ran N1v2) imports it fine.
    """
    import subprocess
    venv_py = REPO / ".venv312" / "Scripts" / "python.exe"
    if not venv_py.exists():
        return False
    try:
        probe = (
            "import sys; sys.path.insert(0, r'%s'); "
            "from qicert.transformers5_compat import install; install(); "
            "import prismatic" % (REPO / "weights" / "code")
        )
        out = subprocess.run(
            [str(venv_py), "-c", probe],
            capture_output=True, text=True, timeout=180,
            cwd=str(REPO),
            env={**os.environ, "PRISMATIC_DATA_ROOT": str(REPO / "weights" / "data")},
        )
        return out.returncode == 0
    except Exception:
        return False


def blueprint_for_seed(seed: int) -> dict:
    """A fully-determined N2 launch blueprint for one seed.  Deterministic:
    everything here can be re-derived from the checkpoint + handoff, so a
    future run reproduces the exact same point set without re-guessing."""
    hs = handoff_status(seed)
    return {
        "seed": seed,
        "handoff": hs,
        "exp_id": "N2",
        "run_tag": f"N2-compression-pareto-seed{seed}",
        "seeds": [seed],
        "backbones": list(BACKBONES),
        "bond_plans": list(BOND_PLANS),
        "layers_scope": LAYERS_SCOPE,
        "ordering": ORDERING,
        "kernel": KERNEL,
        "points": POINTS_PER_SEED,
        "eval_protocol": "matched-budget N1 batch, fine-tuned weights",
        "fine_tuned_handoff_dir": str(HANDOFF),
        "fine_tuned_ckpt_note": (
            "LoRA-merged FT LLM backbone (seed{}.pt) + the EXACT eval batch "
            "(eval_batch_seed{}.pt) + FT reference acc (baseline_seed{}.json). "
            "N2 scores each compressed model on that fixed batch, so delta vs "
            "the FT baseline is a matched-budget comparison.".format(seed, seed, seed)
        ),
        "launch_env_overrides": {
            "PRISMATIC_DATA_ROOT": str(REPO / "weights" / "data"),
            "QICERT_FORK": str(REPO / "weights" / "code"),
            "QICERT_FT_CKPT": str(HANDOFF),
            "QICERT_N2_LAYERS": "all",
            # "QICERT_N2_PLANS" intentionally NOT set here — set it to a
            # sub-window (e.g. "0.50,0.33") when you want a go/no-go on the
            # 2x point first, before the full 6-plan sweep.
        },
        "command_host_venv": (
            "cd \"/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert\" && "
            "export PYTHONUNBUFFERED=1 && "
            "export PRISMATIC_DATA_ROOT=\"$(pwd)/weights/data\" && "
            "export QICERT_FORK=\"$(pwd)/weights/code\" && "
            "export PYTHONPATH=\"$(pwd)/python\" && "
            ".venv312/Scripts/python.exe -m qicert.bench.all --module n2_sweep "
            "--rows=compression-pareto --out results --exp-id N2 --seed {seed} "
            "--run-tag N2-compression-pareto-seed{seed} "
            "--save-ckpt results/N1v2-ckpt --capture heavy "
            "2>&1 | tee results/logs/N2-compression-pareto-seed{seed}.log"
        ).format(seed=seed),
        "command_docker": (
            "cd \"/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert\" && "
            "MSYS_NO_PATHCONV=1 docker run --rm --gpus all "
            "-v \"$(pwd):/workspace\" "
            "-v \"$(pwd)/weights:/workspace/weights\" "
            "-v \"$(pwd)/.hf_cache:/root/.cache/huggingface\" "
            "qicert-dev:cu13-cudaq bash -lc '\n"
            "set -euo pipefail\n"
            "cd /workspace\n"
            "export PYTHONUNBUFFERED=1\n"
            "export PRISMATIC_DATA_ROOT=\"/workspace/weights/data\"\n"
            "export QICERT_FORK=\"/workspace/weights/code\"\n"
            "export QICERT_FT_CKPT=\"/workspace/results/N1v2-ckpt\"\n"
            "export PYTHONPATH=\"/workspace/python\"\n"
            "python3 -m qicert.bench.all --module n2_sweep --rows=compression-pareto "
            "--out results --exp-id N2 --seed {seed} "
            "--run-tag N2-compression-pareto-seed{seed} "
            "--save-ckpt results/N1v2-ckpt --capture heavy "
            "2>&1 | tee results/logs/N2-compression-pareto-seed{seed}.log\n"
            "'"
        ).format(seed=seed),
    }


def write_blueprint(seed: int, blueprint: dict) -> Path:
    N2_DIR.mkdir(parents=True, exist_ok=True)
    p = N2_DIR / f"blueprint_seed{seed}.json"
    p.write_text(json.dumps(blueprint, indent=2))
    return p


def write_pre_launch_snapshot(snap: dict) -> Path:
    N2_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = N2_DIR / "pre_launch_snapshot.json"
    p.write_text(json.dumps(snap, indent=2))
    return p


def write_handoff_index() -> Path:
    """One file summarizing which seeds are clean handoffs now."""
    index = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "handoff_dir": str(HANDOFF),
        "seeds": {str(s): handoff_status(s) for s in (0, 1, 2)},
        "note": (
            "A seed is a CLEAN matched-budget N2 handoff only when seedN.pt "
            "+ eval_batch_seedN.pt + baseline_seedN.json all exist in the "
            "N1v2 handoff dir.  Right now only seed 0 qualifies (the other "
            "N1v2 seed runs were deferred and did not complete with "
            "--save-ckpt results/N1v2-ckpt)."
        ),
    }
    N2_DIR.mkdir(parents=True, exist_ok=True)
    p = N2_DIR / "handoff_index.json"
    p.write_text(json.dumps(index, indent=2))
    return p


def print_launch_command(seed: int, blueprint: dict) -> None:
    """Print the EXACT command to run later, with the per-seed tag already set."""
    bb = blueprint
    print()
    print("=" * 78)
    print(f"N2 — seed {seed} — ready-to-launch command (run LATER, not now)")
    print("=" * 78)
    print()
    print("Pre-condition: seed{} clean handoff = {}".format(seed, bb["handoff"]["clean_handoff"]))
    print("  seed{}.pt            : {}".format(seed, bb["handoff"]["seed_pt_exists"]))
    print("  eval_batch_seed{}.pt : {}".format(seed, bb["handoff"]["eval_batch_exists"]))
    print("  baseline_seed{}.json : {}  (FT ref acc = {})".format(
        seed, bb["handoff"]["baseline_json_exists"], bb["handoff"]["baseline_ft_acc"]))
    print()
    print("If clean handoff is False, do NOT launch N2 for this seed yet — "
          "first run N1v2 for this seed with --save-ckpt results/N1v2-ckpt.")
    print()
    if bb["handoff"]["clean_handoff"]:
        print("Host venv (VERIFIED path — this is what ran N1v2; .venv312 imports "
              "prismatic fine):")
        print()
        print(bb["command_host_venv"])
        print()
        print("Docker (only after `docker build -t qicert-dev:cu13-cudaq -f "
              "Dockerfile .` — image is NOT built on this machine as of "
              "2026-09-12, and Docker Desktop was not running):")
        print()
        print(bb["command_docker"])
        print()
        print("Go/no-go variant (2x point first, smaller wall time) — set "
              "QICERT_N2_PLANS before the chosen command:")
        print()
        print("  export QICERT_N2_PLANS=\"0.50,0.33\"")
        print()
    print("Capture: --out results --exp-id N2 --seed {} --run-tag {} --capture heavy".format(
        seed, bb["run_tag"]))
    print("Per-point artifacts: results/N2/<run_id>/run.json + config.json + system.json + "
          "env.json + metrics.jsonl + ledger.csv row, plus per-layer certificate metrics.")
    print("=" * 78)
    print()


def main() -> None:
    # CPU-only preparation.  No GPU, no VLA load, no dataset.
    snap = pre_launch_snapshot()
    snap_path = write_pre_launch_snapshot(snap)
    handoff_path = write_handoff_index()

    print("N2 preparation complete (CPU-only, no GPU, no VLA, no dataset).")
    print()
    print("Wrote:")
    print("  pre-launch snapshot :", snap_path)
    print("  handoff index       :", handoff_path)
    print()
    print("Seeds with a CLEAN matched-budget N2 handoff right now:")
    for s in (0, 1, 2):
        hs = handoff_status(s)
        flag = "CLEAN" if hs["clean_handoff"] else "NO  "
        print(f"  seed {s}: {flag}  "
              f"seed_pt={hs['seed_pt_exists']}  "
              f"eval_batch={hs['eval_batch_exists']}  "
              f"baseline_json={hs['baseline_json_exists']}  "
              f"FT_ref_acc={hs['baseline_ft_acc']}")
    print()
    print("Full-blown launch would be:")
    for s in (0, 1, 2):
        bp = blueprint_for_seed(s)
        write_blueprint(s, bp)
        if bp["handoff"]["clean_handoff"]:
            print_launch_command(s, bp)
    print()
    print("Blueprints written:")
    for s in (0, 1, 2):
        print("  results/N2/blueprint_seed{}.json".format(s))
    print()
    print("When ready to run, pick a seed with a CLEAN handoff and run the printed "
          "command.  Recommended order:")
    print("  1. seed 0, go/no-go variant (QICERT_N2_PLANS=0.50,0.33), Docker, unbuffered")
    print("  2. if 2x + 0.33 survive, expand to full sweep (QICERT_N2_PLANS unset)")
    print("  3. seeds 1, 2 ONLY after their N1v2 run completes with --save-ckpt "
          "results/N1v2-ckpt (so the handoff trio exists)")
    print()


if __name__ == "__main__":
    main()
