#!/usr/bin/env python3
"""Off-session experiment queue runner for qicert (the GPU pipeline).

Runs every remaining experiment for the submission, one GPU job at a time,
with per-stage pre-flight checks, live timestamped logging, automatic
gating, resume/skip logic, and a final machine-readable summary.

Designed to be launched in a terminal and left alone overnight:

    cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert"
    .venv312/Scripts/python.exe scripts/run_remaining_experiments.py

Useful flags:
    --list              show stages and exit
    --dry-run           run pre-flight checks only (no experiments)
    --exclude n1v2      skip stages (comma-separated: n1v2,calib,n5-plain,...)
    --only n2r2-weighted,n2r2-mixed    run just these stages
    --allow-busy-gpu    don't abort GPU stages when the GPU looks occupied
    --with-tests / --no-tests   full pytest before starting (default: on)

Stages (in order; gates explained inline):
  0 preflight       compile+import every touched module, full pytest
  1 calib-topup     re-collect calibration stats (adds mean|activation|) ~10 min
  2 n2r2-weighted   N2R-v2 activation-weighted residual, FULL-SPLIT (R1 gate)
                    ~1.5-3 h   GATE: GO if some point ratio>=2x and acc>=0.3968
                    (the pre-registered bar delta>=-0.05 vs FT 0.4468)
  3 n2r2-mixed      mixed allocation point (q/k/v/o full precision) ~1 h
                    ONLY runs if stage 2 is NO-GO
  4 n5-plain        degradation curve, uniform truncation plans ~1.5-2 h
  5 n5-weighted     degradation curve, weighted plans ~1 h
  6 n5-predictor    CPU-only predictor fit (seconds) - needs >=5 points total
  7 n1v2            baseline seeds 1 and 2 (18-24 h; runs both in one process,
                    the bench loop clears GPU cache between seeds)
  8 summary         collect every verdict into results/QUEUE-SUMMARY.json

Every stage skips itself if its artifact already exists (so re-running the
script after an interruption continues where it stopped, never repeating a
finished experiment).  A stage whose checkpoint file is a known-bad duplicate
(the pre-2026-09-12 seed1.pt bug) is detected by hash and re-run.

Logs: results/logs/queue/<stage>.log (live, timestamped).  On failure the
stage log's last 40 lines are printed and the queue STOPS (no cascading).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys as _sys
PY = str(_sys.executable)
sys.path.insert(0, str(REPO))
LOGDIR = REPO / "results" / "logs" / "queue"
FT_DIR = REPO / "results" / "N1v2-ckpt"
CALIB = REPO / "results" / "N2R" / "calib_seed0.npz"

_ENV = dict(os.environ)
if "PYTHONPATH" in _ENV:
    _ENV["PYTHONPATH"] = str(REPO / "python") + os.pathsep + str(REPO) + os.pathsep + _ENV["PYTHONPATH"]
else:
    _ENV["PYTHONPATH"] = str(REPO / "python") + os.pathsep + str(REPO)
_ENV["PRISMATIC_DATA_ROOT"] = str(REPO / "weights" / "data")
_ENV["PYTHONUNBUFFERED"] = "1"
ENV_BASE = _ENV


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


# ---------------------------------------------------------------------------
# pre-flight helpers ("double-check the code before each full experiment")
# ---------------------------------------------------------------------------

def preflight_pycompile(modules: list[str]) -> None:
    subprocess.run([PY, "-m", "py_compile", *modules], cwd=REPO,
                   env=ENV_BASE, check=True)


def preflight_import(module: str) -> None:
    subprocess.run([PY, "-c", f"import {module}"], cwd=REPO, env=ENV_BASE,
                   check=True)


def gpu_status() -> tuple[bool, str]:
    """(is_free, human description) via nvidia-smi; True if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        line = out.stdout.strip().splitlines()[0]
        mem, util = (int(x) for x in line.split(","))
        free = mem < 1500 and util < 30
        return free, f"GPU mem {mem} MiB, util {util}%"
    except Exception as exc:
        return True, f"nvidia-smi unavailable ({exc}); assuming free"


def check_ft_checkpoints() -> list[str]:
    """Existence + known-bug detection on the FT handoff checkpoints."""
    problems = []
    seed0 = FT_DIR / "seed0.pt"
    if not seed0.exists():
        problems.append(f"missing {seed0} (run N1 seed 0 first)")
    for s in (1, 2):
        p = FT_DIR / f"seed{s}.pt"
        if p.exists() and seed0.exists():
            if sha256_file(p) == sha256_file(seed0):
                problems.append(
                    f"{p.name} is a byte-identical duplicate of seed0.pt "
                    f"(pre-2026-09-12 --seeds bug) -> stage n1v2 will re-run")
    return problems


# ---------------------------------------------------------------------------
# stage definitions
# ---------------------------------------------------------------------------

def run_cmd(name: str, cmd: list[str], env: dict, expect_minutes: int,
            allow: bool) -> int:
    """Run one command with live timestamped logging. Returns exit code."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / f"{name}.log"
    print(f"\n{'=' * 78}\n=== STAGE {name}  (expected ~{expect_minutes} min)")
    print(f"=== log: {log}\n=== cmd: {' '.join(cmd)}\n{'=' * 78}", flush=True)
    if not allow:
        print("SKIP (gated off by earlier stage / user flag)", flush=True)
        return 0
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            stamp = time.strftime("%H:%M:%S")
            f.write(f"[{stamp}] {line}")
            f.flush()
            print(f"[{stamp}] {line}", end="", flush=True)
        rc = proc.wait()
    dt = time.time() - t0
    print(f"\n=== STAGE {name} finished rc={rc} in {fmt_dur(dt)} "
          f"(expected ~{expect_minutes} min)\n", flush=True)
    if rc != 0:
        print(f"!!! stage {name} FAILED — last 40 log lines:", flush=True)
        lines = open(log, encoding="utf-8", errors="replace").readlines()
        print("".join(lines[-40:]), flush=True)
    return rc


def n2r2_verdict() -> dict:
    """Best completed full-split N2R2 weighted point -> GO/NO-GO + details."""
    best = None
    root = REPO / "results" / "N2R2"
    if root.exists():
        for rj in sorted(root.glob("*/run.json"), reverse=True):
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            cfg, res = r.get("config", {}), r.get("results", {})
            if (cfg.get("fit_mode") == "weighted" and r.get("status") == "completed"
                    and "full-split" in str(res.get("eval_scope", ""))
                    and cfg.get("mixed_allocation") is not True):
                ratio, acc = res.get("ratio"), res.get("eval_acc")
                if ratio and acc and acc == acc and (best is None
                                                     or acc > best["acc"]):
                    best = {"ratio": float(ratio), "acc": float(acc),
                            "run": str(rj.parent.name)}
    go = bool(best and best["ratio"] >= 2.0
              and best["acc"] >= 0.4468 - 0.05)
    return {"go": go, "best": best,
            "bar": "ratio>=2.0 and acc>=0.3968 (delta>=-0.05 vs FT 0.4468)"}


# artifact-completion tests --------------------------------------------------

def done_calib() -> bool:
    if not CALIB.exists():
        return False
    try:
        code = ("import sys; sys.path.insert(0, r'%s'); "
                "from qicert.calibrate import load_stats; "
                "s = load_stats(r'%s'); "
                "v = s[next(iter(s))]; "
                "print('MEAN_OK' if 'mean_abs' in v else 'MEAN_MISSING')"
                % (REPO / "python", CALIB))
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True)
        return "MEAN_OK" in out.stdout
    except Exception:
        return False


def done_n2r2_weighted() -> bool:
    return n2r2_verdict()["best"] is not None


def done_n2r2_mixed() -> bool:
    root = REPO / "results" / "N2R2"
    if not root.exists():
        return False
    for rj in sorted(root.glob("*/run.json"), reverse=True):
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        cfg = r.get("config", {})
        if (cfg.get("mixed_allocation") is True and r.get("status") == "completed"
                and "full-split" in str(r.get("results", {}).get("eval_scope", ""))):
            return True
    return False


def done_n5_points(min_points: int) -> bool:
    f = REPO / "results" / "lyapunov" / "degradation_seed0.json"
    if not f.exists():
        return False
    try:
        pts = json.loads(f.read_text()).get("points", [])
        return len(pts) >= min_points
    except Exception:
        return False


def done_n1v2_seeds() -> bool:
    s0, s1, s2 = (FT_DIR / f"seed{s}.pt" for s in (0, 1, 2))
    if not (s0.exists() and s1.exists() and s2.exists()):
        return False
    return (sha256_file(s1) != sha256_file(s0)
            and sha256_file(s2) != sha256_file(s0))


BENCH = [PY, "-m", "bench.all"]
import bench.n2r_search

# ---- confirmation candidate selection (reads search results) --------------

def _search_candidate_dirs():
    root = REPO / "results" / "N2R-search"
    if not root.exists():
        return []
    return sorted(root.glob("*"))


def _candidate_provgo(dir_path: Path):
    try:
        j = json.loads((dir_path / "run.json").read_text())
    except Exception:
        return False
    r = j.get("results", {})
    return bool(r.get("provisional_go"))


def _candidate_best_acc(dir_path: Path):
    try:
        j = json.loads((dir_path / "run.json").read_text())
    except Exception:
        return None
    r = j.get("results", {})
    return r.get("search_acc")


def _confirmation_candidate():
    """Pick one searched configuration to confirm, with a deterministic tie-break.

    Preference order:
      1. a provisional-GO candidate with the highest search acc
      2. otherwise the highest-search-acc candidate overall
    If nothing was searched yet, return None.
    """
    dirs = _search_candidate_dirs()
    if not dirs:
        return None
    go = [d for d in dirs if _candidate_provgo(d)]
    pool = go if go else dirs
    best = max(pool, key=_candidate_best_acc)
    try:
        j = json.loads((best / "run.json").read_text())
    except Exception:
        return None
    cfg = j.get("config", {})
    return {
        "plan": str(cfg.get("plan_fraction")),
        "rrank": str(cfg.get("residual_rank_cap")),
        "mode": str(cfg.get("fit_mode", "weighted")),
        "mixed": str(cfg.get("mixed_allocation", False)),
        "stat": str(cfg.get("calib_stat", "mean") or "mean"),
        "tt_split": str(cfg.get("tt_split", 0.6)),
        "source_dir": str(best),
    }


def _confirmation_done():
    root = REPO / "results" / "N2R2"
    if not root.exists():
        return False
    for rj in sorted(root.glob("*/run.json"), reverse=True):
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        cfg = r.get("config", {})
        res = r.get("results", {})
        if (cfg.get("fit_mode") == _confirm_mode()
                and cfg.get("plan_fraction") == float(_confirm_plan())
                and cfg.get("residual_rank_cap") == int(_confirm_rrank())
                and cfg.get("mixed_allocation") == (_confirm_mixed() == "1")
                and r.get("status") == "completed"
                and "full-split" in str(res.get("eval_scope", ""))):
            return True
    return False


def _confirm_mode():
    c = _confirmation_candidate()
    return c["mode"] if c else "weighted"


def _confirm_stat():
    c = _confirmation_candidate()
    return c["stat"]


def _confirm_plan():
    c = _confirmation_candidate()
    return c["plan"]


def _confirm_rrank():
    c = _confirmation_candidate()
    return c["rrank"]


def _confirm_mixed():
    c = _confirmation_candidate()
    return c["mixed"]


STAGES: list[dict] = [
    {
        "name": "preflight",
        "expect_min": 5,
        "allow": lambda a: True,
        "done": lambda: False,
        "cmd": lambda: None,          # handled specially
    },
    {
        "name": "calib-topup",
        "expect_min": 10,
        "allow": lambda a: "calib" not in a.exclude,
        "done": lambda: done_calib(),
        "cmd": lambda: ([PY, "-m", "qicert.calibrate"], ENV_BASE),
    },    {
        "name": "n2r2-search",
        "expect_min": 40,
        "allow": lambda a: "n2r2-search" not in a.exclude,
        "done": lambda: (REPO / "results" / "N2R-search").exists(),
        "cmd": lambda: (
            BENCH + ["--module", "n2r_search", "--rows=n2r-search",
                     "--out", "results", "--exp-id", "N2R2",
                     "--run-tag", "N2R2-search"],
            {**ENV_BASE, "QICERT_N2_EVAL": "full",
             "QICERT_N2R2_MODE": "weighted", "QICERT_N2R2_STAT": "mean",
             "QICERT_N2R2_PLANS": "0.50,0.33",
             "QICERT_N2R2_RRANKS": "32,64",
             "QICERT_N2R2_MIXED": "1",              "QICERT_N2R_TTSPLIT": "0.6",
             "QICERT_N2R_SEARCH_BATCHES": "40",
             "QICERT_N2R_SEARCH_GO": "0.30",
             "QICERT_FT_CKPT": str(FT_DIR)}),
    },
    {
        "name": "n2r2-confirm",
        "expect_min": 180,
        "allow": lambda a: ("n2r2-confirm" not in a.exclude
                            and _confirmation_candidate() is not None),
        "done": lambda: _confirmation_done(),
        "cmd": lambda: (
            BENCH + ["--module", "n2r_search", "--rows=n2r-confirm",
                     "--out", "results", "--exp-id", "N2R2",
                     "--run-tag", "N2R2-confirm"],
            {**ENV_BASE, "QICERT_N2_EVAL": "full",
             "QICERT_N2R2_MODE": _confirm_mode(),
             "QICERT_N2R2_STAT": _confirm_stat(),
             "QICERT_N2R2_PLANS": _confirm_plan(),
             "QICERT_N2R2_RRANKS": _confirm_rrank(),
             "QICERT_N2R2_MIXED": "1" if _confirm_mixed() else "0",
             "QICERT_N2R_TTSPLIT": "0.6",
             "QICERT_FT_CKPT": str(FT_DIR)}),
    },
    {
        "name": "n2r2-weighted",
        "expect_min": 180,
        "allow": lambda a: ("n2r2-weighted" not in a.exclude
                            and not (REPO / "results" / "N2R-search").exists()
                            and not _confirmation_candidate()),
        "done": lambda: done_n2r2_weighted(),
        "cmd": lambda: (
            BENCH + ["--module", "n2r2_sweep", "--rows=n2r2-go-no-go",
                     "--out", "results", "--exp-id", "N2R2", "--seed", "0",
                     "--run-tag", "N2R2-weighted-go-no-go", "--capture", "heavy"],
            {**ENV_BASE, "QICERT_N2_EVAL": "full",
             "QICERT_N2R2_MODE": "weighted", "QICERT_N2R2_STAT": "absmax",
             "QICERT_N2R2_PLANS": "0.50", "QICERT_N2R2_RRANKS": "32,64",
             "QICERT_FT_CKPT": str(FT_DIR)}),
    },
    {
        "name": "n2r2-mixed",
        "expect_min": 60,
        "allow": lambda a: ("n2r2-mixed" not in a.exclude
                            and not _confirmation_candidate()
                            and not n2r2_verdict()["go"]),
        "done": lambda: done_n2r2_mixed(),
        "cmd": lambda: (
            BENCH + ["--module", "n2r2_sweep", "--rows=n2r2-go-no-go",
                     "--out", "results", "--exp-id", "N2R2", "--seed", "0",
                     "--run-tag", "N2R2-mixed-go-no-go", "--capture", "heavy"],
            {**ENV_BASE, "QICERT_N2_EVAL": "full",
             "QICERT_N2R2_MODE": "weighted", "QICERT_N2R2_STAT": "absmax",
             "QICERT_N2R2_MIXED": "1", "QICERT_N2R2_PLANS": "0.50",
             "QICERT_N2R2_RRANKS": "32", "QICERT_FT_CKPT": str(FT_DIR)}),
    },
    {
        "name": "n5-plain",
        "expect_min": 110,
        "allow": lambda a: "n5-plain" not in a.exclude,
        "done": lambda: done_n5_points(3),
        "cmd": lambda: (
            BENCH + ["--module", "lyapunov_curve", "--rows=lyapunov-degradation",
                     "--out", "results", "--exp-id", "N5", "--seed", "0",
                     "--run-tag", "N5-degradation-plain", "--capture", "heavy"],
            {**ENV_BASE, "QICERT_N5_PLANS": "0.50,0.33,0.25",
             "QICERT_N5_RRANKS": "32", "QICERT_N5_MODE": "plain",
             "QICERT_N5_BATCHES": "500",
             "QICERT_N2_EVAL": "full",
             "QICERT_FT_CKPT": str(FT_DIR)}),
    },
    {
        "name": "n5-weighted",
        "expect_min": 70,
        "allow": lambda a: "n5-weighted" not in a.exclude,
        "done": lambda: done_n5_points(5),
        "cmd": lambda: (
            BENCH + ["--module", "lyapunov_curve", "--rows=lyapunov-degradation",
                     "--out", "results", "--exp-id", "N5", "--seed", "0",
                     "--run-tag", "N5-degradation-weighted", "--capture", "heavy"],
            {**ENV_BASE, "QICERT_N5_PLANS": "0.50,0.33",
             "QICERT_N5_RRANKS": "32", "QICERT_N5_MODE": "weighted",
             "QICERT_N5_BATCHES": "500",
             "QICERT_N2_EVAL": "full",
             "QICERT_FT_CKPT": str(FT_DIR)}),
    },
    {
        "name": "n5-predictor",
        "expect_min": 1,
        "allow": lambda a: ("n5-predictor" not in a.exclude
                            and done_n5_points(3)),
        "done": lambda: (REPO / "results" / "lyapunov" / "predictor.json").exists(),
        "cmd": lambda: (
            BENCH + ["--module", "lyapunov_curve", "--rows=lyapunov-predictor",
                     "--out", "results", "--exp-id", "N5", "--seed", "0",
                     "--run-tag", "N5-predictor"], ENV_BASE),
    },
    {
        "name": "n1v2",
        "expect_min": 1200,
        "allow": lambda a: "n1v2" not in a.exclude,
        "done": lambda: done_n1v2_seeds(),
        "cmd": lambda: (
            BENCH + ["--module", "compress", "--rows", "baseline-int8",
                     "--out", "results", "--exp-id", "N1v2",
                     "--seeds", "1,2", "--steps-per-seed", "1200",
                     "--batch", "2", "--save-ckpt", str(FT_DIR),
                     "--run-tag", "host-seeds12"],
            ENV_BASE),
    },
]


def stage_preflight(stage: dict) -> None:
    """Per-stage code double-checks (compile + import)."""
    m = stage["name"]
    if m in ("calib-topup",):
        preflight_pycompile(["python/qicert/calibrate.py"])
        preflight_import("qicert.calibrate")
    elif m == "n2r2-search":
        preflight_pycompile(["bench/n2r_search.py", "bench/n2r2_sweep.py",
                             "bench/n2_sweep.py",
                             "python/qicert/compress_residual.py"])
        preflight_import("bench.n2r_search")
    elif m == "n2r2-confirm":
        preflight_pycompile(["bench/n2r_search.py", "bench/n2r2_sweep.py",
                             "bench/n2_sweep.py",
                             "python/qicert/compress_residual.py"])
        preflight_import("bench.n2r_search")
    elif m in ("n2r2-weighted", "n2r2-mixed"):
        preflight_pycompile(["bench/n2r2_sweep.py", "bench/n2_sweep.py",
                             "python/qicert/compress_residual.py"])
        preflight_import("bench.n2r2_sweep")
    elif m in ("n5-plain", "n5-weighted", "n5-predictor"):
        preflight_pycompile(["bench/lyapunov_curve.py",
                             "python/qicert/compress_residual.py"])
        preflight_import("bench.lyapunov_curve")
    elif m == "n1v2":
        preflight_pycompile(["bench/compress.py", "bench/all.py"])
        preflight_import("bench.compress")
    if m not in ("preflight",):
        probs = check_ft_checkpoints()
        if probs and m != "n1v2":
            print(f"  [preflight:{m}] WARNING: " + "; ".join(probs), flush=True)


def run_preflight(args) -> bool:
    print("=== PREFLIGHT: compile + import all touched modules", flush=True)
    mods = ["python/qicert/calibrate.py", "python/qicert/compress_residual.py",
            "python/qicert/safety.py", "bench/n2r2_sweep.py",
            "bench/n2r_search.py", "bench/lyapunov_curve.py", "bench/safety.py",
            "bench/compiler.py", "bench/monitor.py", "bench/latency.py",
            "bench/all.py"]
    preflight_pycompile(mods)
    for mod in ("bench.n2r2_sweep", "bench.n2r_search",
                "bench.lyapunov_curve", "bench.safety",
                "bench.compiler", "bench.monitor", "bench.latency"):
        preflight_import(mod)
    print("    compile+import OK", flush=True)
    free, desc = gpu_status()
    print(f"    GPU: {desc}", flush=True)
    if not free and not args.allow_busy_gpu:
        print("    GPU looks BUSY — aborting (use --allow-busy-gpu to override)",
              flush=True)
        return False
    probs = check_ft_checkpoints()
    for p in probs:
        print(f"    checkpoint check: {p}", flush=True)
    if args.with_tests:
        print("=== PREFLIGHT: full pytest (80 tests, ~3 min)", flush=True)
        rc = subprocess.run([PY, "-m", "pytest", "tests/", "-q"], cwd=REPO,
                            env=ENV_BASE).returncode
        if rc != 0:
            print("    TESTS FAILED — aborting queue", flush=True)
            return False
    return True


def run_summary() -> None:
    v = n2r2_verdict()
    n5p = REPO / "results" / "lyapunov" / "predictor.json"
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n2r2_weighted": v,
        "n1v2_seeds_ok": done_n1v2_seeds(),
        "n5_points": len(json.loads(
            (REPO / "results" / "lyapunov" / "degradation_seed0.json").read_text()
        ).get("points", [])) if (REPO / "results" / "lyapunov"
                                 / "degradation_seed0.json").exists() else 0,
        "n5_predictor": json.loads(n5p.read_text()) if n5p.exists() else None,
    }
    dest = REPO / "results" / "QUEUE-SUMMARY.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(f"\n{'=' * 78}\n=== QUEUE SUMMARY -> {dest}\n{'=' * 78}")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--with-tests", dest="with_tests", action="store_true",
                    default=True)
    ap.add_argument("--no-tests", dest="with_tests", action="store_false")
    args = ap.parse_args()

    names = [s["name"] for s in STAGES]
    if args.list:
        for s in STAGES:
            print(f"{s['name']:<16} ~{s['expect_min']:>5} min")
        return 0
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        for s in STAGES:
            if s["name"] not in want:
                s["allow"] = lambda a, _n=s["name"]: False

    args.exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    t_start = time.time()
    print(f"qicert experiment queue — start {time.strftime('%H:%M:%S')}\n"
          f"repo: {REPO}\ninterpreter: {PY}\n", flush=True)

    if not run_preflight(args):
        return 2

    # ---- search config summary before the search stage starts ---------------
    print("\n--- N2R-search configuration plan", flush=True)
    for cfg in bench.n2r_search._search_grid():
        print("    candidate:", json.dumps(cfg, indent=2), flush=True)
    print("    search budget:",
          ENV_BASE.get("QICERT_N2R_SEARCH_BATCHES", "40 (stage default)"),
          "batches", flush=True)
    print("    provisional GO bar:",
          ENV_BASE.get("QICERT_N2R_SEARCH_GO", "0.30 (stage default)"),
          "search acc", flush=True)
    print("    rules:",
          "fixed budget, cached ref preds, conservative early stopping,",
          "search != reported number", flush=True)

    ok = True
    for stage in STAGES:
        name = stage["name"]
        if not stage["allow"](args):
            print(f"\n--- stage {name}: SKIPPED (gate/flag)", flush=True)
            continue
        if stage["done"]():
            print(f"\n--- stage {name}: already complete — skipped "
                  f"(delete its artifacts or use --only to force)", flush=True)
            continue
        if args.dry_run:
            print(f"\n--- stage {name}: DRY-RUN (would execute, "
                  f"~{stage['expect_min']} min)", flush=True)
            continue
        stage_preflight(stage)
        if name == "preflight":
            continue
        cmd, env = stage["cmd"]()
        rc = run_cmd(name, cmd, env, stage["expect_min"], allow=True)
        if rc != 0:
            ok = False
            print(f"\n!!! queue STOPPED at stage {name} (rc={rc}). "
                  f"Fix, then re-run this script — finished stages auto-skip.",
                  flush=True)
            break
        if name == "n2r2-search":
            cand = _confirmation_candidate()
            if cand:
                print(f"\n=== N2R-search CANDIDATE FOR CONFIRMATION:", flush=True)
                print(json.dumps(cand, indent=2), flush=True)
            else:
                print(f"\n=== N2R-search: no candidate selected (no search results)",
                      flush=True)
        if name == "n2r2-confirm":
            cand = _confirmation_candidate()
            print(f"\n=== N2R-confirm target: {cand}", flush=True)
        if name == "n2r2-weighted":
            v = n2r2_verdict()
            print(f"\n=== N2R-v2 GATE: {'GO' if v['go'] else 'NO-GO'} "
                  f"({v['bar']}; best={v['best']})", flush=True)

    run_summary()
    print(f"\nqueue {'COMPLETE' if ok else 'STOPPED EARLY'} in "
          f"{fmt_dur(time.time() - t_start)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
