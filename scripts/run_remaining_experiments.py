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

Stages (in order; gates explained inline; times measured on RTX 5060,
superseding the stale Kaggle-T4 estimates):
  0 preflight       compile+import every touched module, full pytest (~4 min)
  1 calib-topup     re-collect calibration stats (adds mean|activation|) ~3 min
  2 n2r2-search     cheap 40-batch-eval config search over 4-5 candidates ~1 h
  3 n2r2-confirm    full-protocol run of the search-selected config ~30 min
                    GATE (the pre-registered accuracy bar):
                    GO if some full-split point ratio>=2x and acc>=0.3968
                    (delta>=-0.05 vs FT 0.4468)
  4 n2r2-weighted   fallback: two full points (only if no search artifacts)
  5 n2r2-mixed      fallback: one mixed point (only if no search artifacts)
  6 n5-plain        degradation curve, uniform truncation plans ~45 min
  7 n5-weighted     degradation curve, weighted plans ~30 min
  8 n5-predictor    CPU-only predictor fit (seconds) - needs >=3 points total
  9 n1v2            baseline seeds 1 and 2 (~1-2 h; runs both in one process,
                    the bench loop clears GPU cache between seeds)
 10 n9-repair      post-compression LoRA repair training (~1.5 h; N2R2
                    confirmed config, then r=8 LoRA x 1200 steps on
                    train-only episodes; prefix evals + cert bookkeeping)
 11 n10-scale      Qwen2.5-1.5B backbone compression probe (~1-1.5 h +
                    ~3 GB HF download; CPU-only weight-space measurement,
                    no task-accuracy claim - see PHASE2-CASE.md)
 12 n9-confirm     N9 full-protocol confirm (~30 min; full 6496-batch eval
                    of saved merged weights + re-derived dense certs +
                    adapter-counted ratio; the REPORTABLE N9 row)
 13 summary         collect every verdict into results/QUEUE-SUMMARY.json

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
    """Best completed full-split N2R2 weighted point -> GO/NO-GO + details.

    Schema note (audit 2026-09-13): RunRecorder stores the config in the
    sibling config.json and mirrors its fields inside run.json["results"];
    run.json has NO top-level "config" key (earlier code read one).
    """
    best = None
    root = REPO / "results" / "N2R2"
    if root.exists():
        for rj in sorted(root.glob("*/run.json"), reverse=True):
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            res = r.get("results", {}) or {}
            if (res.get("fit_mode") == "weighted"
                    and r.get("status") == "completed"
                    and "full-split" in str(res.get("eval_scope", ""))
                    and res.get("mixed_allocation") is not True):
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
        res = r.get("results", {}) or {}
        if (res.get("mixed_allocation") is True
                and r.get("status") == "completed"
                and "full-split" in str(res.get("eval_scope", ""))):
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


def _exp_complete(exp_id: str) -> bool:
    """True once any run dir for this exp-id holds a completed run.json."""
    import glob as _glob
    import json as _json
    for rj in _glob.glob(str(REPO / "results" / exp_id / "*" / "run.json")):
        try:
            if _json.loads(Path(rj).read_text()).get("status") == "completed":
                return True
        except Exception:
            continue
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
    """Run dirs holding search rows.

    Audit 2026-09-13: the search stage runs under --exp-id N2R2, so its
    rows land in results/N2R2/ (the recorder keys dirs by experiment id),
    NOT in results/N2R-search/.  Search rows are identified by carrying
    results.search_acc (only _run_search writes that field); this also
    keeps full-protocol confirm rows (no search_acc, full-split scope)
    from being mistaken for candidates.
    """
    dirs: list[Path] = []
    for root in (REPO / "results" / "N2R2", REPO / "results" / "N2R-search"):
        if not root.exists():
            continue
        for rj in sorted(root.glob("*/run.json")):
            try:
                j = json.loads(rj.read_text())
            except Exception:
                continue
            if "search_acc" in (j.get("results", {}) or {}):
                dirs.append(rj.parent)
    return sorted(set(dirs))


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
        acc = j.get("results", {}).get("search_acc")
    except Exception:
        return -1.0
    if acc is None or acc != acc:          # None or NaN
        return -1.0
    return float(acc)


def _search_has_completed_candidate() -> bool:
    """Stage n2r2-search is done only if >=1 candidate recorded a usable run.

    2026-09-13: the first search launch had every candidate FAIL at runtime
    (ModuleNotFoundError) yet the stage exited rc=0 and wrote no candidate
    dirs — the queue then 'completed' while achieving nothing.  done() now
    requires a selectable candidate, and main() hard-fails the queue when
    the search produced none (see the n2r2-search post-check).
    """
    return _confirmation_candidate() is not None


def _confirmation_candidate():
    """Pick one searched configuration to confirm, with a deterministic tie-break.

    Preference order:
      1. a provisional-GO candidate with the highest search acc
      2. otherwise the highest-search-acc candidate overall
    If nothing was searched yet, return None.
    """
    # 2026-09-13: absmax decisively beats mean on this backbone (search arms
    # 0.031-0.122; full-split absmax 0.164; the interrupted mean-stat confirm
    # was heading to ~0.04).  If a completed absmax full-split point exists,
    # prefer it over the search-selected config instead of re-confirming a
    # known-worse region of the grid.
    root = REPO / "results" / "N2R2"
    if root.exists():
        for rj in sorted(root.glob("*/run.json"), reverse=True):
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            res = r.get("results", {}) or {}
            if (res.get("fit_mode") == "weighted"
                    and (res.get("calib_stat") or "mean") == "absmax"
                    and res.get("mixed_allocation") is not True
                    and r.get("status") == "completed"
                    and "full-split" in str(res.get("eval_scope", ""))
                    and res.get("plan_fraction") is not None):
                return {
                    "plan": str(res.get("plan_fraction")),
                    "rrank": str(res.get("residual_rank_cap")),
                    "mode": "weighted",
                    "mixed": False,
                    "stat": "absmax",
                    "tt_split": str(res.get("tt_split", 0.6)),
                    "source_dir": str(rj.parent),
                }
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
    # Search runs store the candidate config inside results["config"];
    # the sibling config.json carries the same dict under {"config": ...}.
    cfg = (j.get("results", {}) or {}).get("config")
    if not cfg:
        try:
            cfg = json.loads((best / "config.json").read_text()).get("config", {})
        except Exception:
            cfg = {}
    return {
        "plan": str(cfg.get("plan_fraction")),
        "rrank": str(cfg.get("residual_rank_cap")),
        "mode": str(cfg.get("fit_mode", "weighted")),
        "mixed": bool(cfg.get("mixed_allocation", False)),
        "stat": str(cfg.get("calib_stat") or "mean"),
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
        res = r.get("results", {}) or {}
        if (res.get("fit_mode") == _confirm_mode()
                and (res.get("calib_stat") or "mean") == _confirm_stat()
                and res.get("plan_fraction") == float(_confirm_plan())
                and res.get("residual_rank_cap") == int(_confirm_rrank())
                and bool(res.get("mixed_allocation")) == bool(_confirm_mixed())
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
    },    {
        "name": "calib-topup",
        "expect_min": 3,   # measured 2.6 min on RTX 5060 (2026-09-12 queue log)
        "allow": lambda a: "calib" not in a.exclude,
        "done": lambda: done_calib(),
        "cmd": lambda: ([PY, "-m", "qicert.calibrate"], ENV_BASE),
    },
    {
        "name": "n2r2-search",
        "expect_min": 60,  # ~12 min compress per candidate x 4-5 + seconds eval
        "allow": lambda a: "n2r2-search" not in a.exclude,
        "done": lambda: _search_has_completed_candidate(),
        "cmd": lambda: (
            BENCH + ["--module", "n2r_search", "--rows=n2r-search",
                     "--out", "results", "--exp-id", "N2R2",
                     "--run-tag", "N2R2-search"],
            {**ENV_BASE, "QICERT_N2_EVAL": "full",
             # stat: absmax, NOT mean -- evidence 2026-09-13: every mean-stat
             # arm measured erratic/bad (search 0.031-0.122; full-split
             # confirm interrupted at acc~0.04) while the pre-registered
             # absmax runs measured 0.164 full-split.  absmax concentrates
             # residual correction on outlier channels; mean dilutes it.
             "QICERT_N2R2_MODE": "weighted", "QICERT_N2R2_STAT": "absmax",
             "QICERT_N2R2_PLANS": "0.50,0.33",
             "QICERT_N2R2_RRANKS": "32,64",
             "QICERT_N2R2_MIXED": "1",              "QICERT_N2R_TTSPLIT": "0.6",
             "QICERT_N2R_SEARCH_BATCHES": "40",
             "QICERT_N2R_SEARCH_GO": "0.30",
             "QICERT_FT_CKPT": str(FT_DIR)}),
    },
    {
        "name": "n2r2-confirm",
        "expect_min": 30,  # measured full point ~27 min (11.5 compress + 15 eval)
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
        "expect_min": 60,  # fallback: 2 full points x ~27 min (stale T4 est. was 180)
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
        "expect_min": 30,  # fallback: 1 full point ~27 min
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
        "expect_min": 45,  # 3 plans x (~12 min compress + ~2 min 500-batch preds)
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
        "expect_min": 30,  # 2 plans, ref cache reused
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
        # Measured on 5060 (N1v2 seed-0 log): ~8.5 min train + ~20.5 min full eval
        # per seed -> ~1-2 h for seeds 1+2 (stale T4 est. was 1200 min).
        "expect_min": 120,
        "allow": lambda a: "n1v2" not in a.exclude,
        "done": lambda: done_n1v2_seeds(),
        "cmd": lambda: (
            BENCH + ["--module", "compress", "--rows", "baseline-int8",
                     "--out", "results", "--exp-id", "N1v2",
                     "--seeds", "1,2", "--steps-per-seed", "1200",
                     "--batch", "2",                     "--save-ckpt", str(FT_DIR),
                     "--run-tag", "host-seeds12"],
            ENV_BASE),
    },
    {
        "name": "n9-repair",
        # Measured components on 5060: compress ~11.5 min + 2 prefix evals
        # (~3 min at 600 batches) + 1200 LoRA steps (~55-70 min) + save.
        "expect_min": 100,
        "allow": lambda a: "n9-repair" not in a.exclude,
        "done": lambda: _exp_complete("N9"),
        "cmd": lambda: (
            BENCH + ["--module", "n9_repair", "--rows", "repair",
                     "--out", "results", "--exp-id", "N9",
                     "--run-tag", "n9-repair-frac0.50-rr64"],
            {**ENV_BASE, "QICERT_FT_CKPT": str(FT_DIR),
             "QICERT_N2_EVAL": "full"}),
    },
    {
        "name": "n9-confirm",
        # Eval-only: VLA load ~1.5 min + full 6496-batch eval ~20 min +
        # 168 dense SVDs on CPU (~2 min). No training.
        "expect_min": 30,
        "allow": lambda a: ("n9-confirm" not in a.exclude
                            and (REPO / "results" / "N9-repaired"
                                 / "repaired_seed0.pt").exists()),
        "done": lambda: _exp_complete("N9C"),
        "cmd": lambda: (
            BENCH + ["--module", "n9_repair", "--rows", "repair-confirm",
                     "--out", "results", "--exp-id", "N9C", "--seed", "0",
                     "--run-tag", "n9-confirm-seed0", "--capture", "heavy"],
            {**ENV_BASE, "QICERT_FT_CKPT": str(FT_DIR),
             "QICERT_N2_EVAL": "full"}),
    },
    {
        "name": "n10-scale",
        # CPU-only (SVD work + state dicts in system RAM); first launch adds
        # a ~3 GB HF download. Two full SVD passes per layer per plan at
        # 1.5B shapes (~15 s/layer) -> ~1.5-2 h for 196 layers x 2 plans.
        "expect_min": 110,
        "allow": lambda a: "n10-scale" not in a.exclude,
        "done": lambda: _exp_complete("N10"),
        "cmd": lambda: (
            BENCH + ["--module", "n10_scale", "--rows", "scale-probe",
                     "--out", "results", "--exp-id", "N10",
                     "--run-tag", "n10-qwen25-1_5b"],
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
    elif m in ("n9-repair", "n9-confirm"):
        preflight_pycompile(["bench/n9_repair.py", "bench/n2_sweep.py",
                             "python/qicert/compress_residual.py"])
        preflight_import("bench.n9_repair")
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
                # 2026-09-13: all candidates failed (or produced no usable
                # rows) -> the queue must NOT report success.  The stage log
                # has the per-candidate tracebacks.
                ok = False
                print(f"\n!!! n2r2-search produced NO usable candidate "
                      f"(all candidates failed or empty). See "
                      f"results/logs/queue/n2r2-search.log — queue marked "
                      f"FAILED.", flush=True)
                break
        if name == "n2r2-confirm":
            cand = _confirmation_candidate()
            print(f"\n=== N2R-confirm target: {cand}", flush=True)
            v = n2r2_verdict()
            print(f"\n=== N2R2 GATE after confirm: "
                  f"{'GO' if v['go'] else 'NO-GO'} ({v['bar']}; "
                  f"best={v['best']})", flush=True)
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
