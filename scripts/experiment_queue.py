#!/usr/bin/env python
"""qicert sequential experiment queue — one GPU experiment at a time.

Runs each queued bench invocation to completion, then starts the next one.
No parallel GPU jobs (the 2026-09-11 host-freeze lesson: two parallel N1v2
runs locked the machine). Progress is heavily logged with timestamps so the
operator can tail one file and always know where the queue is.

Pipeline (verdict-based branching):
    1. wait for the already-running go/no-go run to finish
    2. summarize it -> GO  = some completed point at ratio >= 2x is within
                            5% absolute drop of the FT reference (R1 bar)
       NO-GO               = stop the compression track; log the honest
                             negative result; do NOT launch the full sweep
    3. if GO -> full 12-point N2 sweep (seed 0, same protocol)
    4. summarize the full sweep -> update the experiment log

Usage:
    python scripts/experiment_queue.py --log results/logs/queue.log
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv312" / "Scripts" / "python.exe"
LOG_DIR = REPO / "results" / "logs"
N2_DIR = REPO / "results" / "N2"
HANDOFF = REPO / "results" / "N1v2-ckpt"

GONOGO_TAG = "N2-go-no-go-seed0"
FULL_TAG = "N2-compression-pareto-seed0"
GONOGO_LOG = LOG_DIR / f"{GONOGO_TAG}.log"
FULL_LOG = LOG_DIR / f"{FULL_TAG}.log"
GONOGO_VERDICT = N2_DIR / "verdict-go-no-go.json"
FULL_VERDICT = N2_DIR / "verdict-full.json"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def bench_env(plans: str | None) -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "PRISMATIC_DATA_ROOT": str(REPO / "weights" / "data"),
        "QICERT_FORK": str(REPO / "weights" / "code"),
        "QICERT_FT_CKPT": str(HANDOFF),
        "PYTHONPATH": str(REPO / "python"),
    })
    if plans:
        env["QICERT_N2_PLANS"] = plans
    else:
        env.pop("QICERT_N2_PLANS", None)
    return env


def bench_cmd(tag: str) -> list[str]:
    return [
        str(PY), "-m", "qicert.bench.all",
        "--module", "n2_sweep", "--rows=compression-pareto",
        "--out", "results", "--exp-id", "N2", "--seed", "0",
        "--run-tag", tag, "--save-ckpt", "results/N1v2-ckpt",
        "--capture", "heavy",
    ]


def summarize(tag: str, verdict_json: Path) -> dict:
    out_md = N2_DIR / "summary.md"
    r = subprocess.run(
        [str(PY), "scripts/n2_summarize.py",
         "--results-root", "results", "--exp-id", "N2",
         "--invocation-tag", tag,
         "--out", str(out_md), "--verdict-json", str(verdict_json)],
        cwd=str(REPO), capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        log(f"summarize failed rc={r.returncode}: {r.stderr[-500:]}")
        return {}
    try:
        return json.loads(verdict_json.read_text())
    except Exception as exc:
        log(f"verdict json unreadable: {exc}")
        return {}


def run_and_wait(cmd: list[str], env: dict, log_path: Path) -> int:
    """Run a bench invocation to completion, streaming output into log_path."""
    log(f"LAUNCH: {' '.join(cmd)}")
    log(f"env: QICERT_N2_PLANS={env.get('QICERT_N2_PLANS', '<all plans>')}")
    t0 = time.time()
    with open(log_path, "a") as lf:
        lf.write(f"\n===== queue launch {datetime.now().isoformat()} =====\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                stdout=lf, stderr=subprocess.STDOUT)
        log(f"PID {proc.pid}; tail -f {log_path}")
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(20)
    dt = time.time() - t0
    log(f"EXIT rc={rc} after {dt/60:.1f} min ({log_path.name})")
    return rc


def bench_running() -> bool:
    """True if any qicert.bench.all python process is alive (Windows-safe)."""
    try:
        out = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe'", "get", "commandline"],
            capture_output=True, text=True, timeout=30)
        return "qicert.bench.all" in (out.stdout or "")
    except Exception:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter "
                 "\"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"],
                capture_output=True, text=True, timeout=60)
            return "qicert.bench.all" in (out.stdout or "")
        except Exception:
            return False


def wait_for_go_no_go() -> None:
    if not bench_running():
        log("no bench process alive; go/no-go run assumed finished")
        return
    log(f"go/no-go run still in progress; waiting (log: {GONOGO_LOG.name})")
    while bench_running():
        time.sleep(30)
    log("go/no-go run finished")


def main() -> int:
    log("=== qicert experiment queue started ===")
    log(f"step 1: wait for go/no-go ({GONOGO_TAG})")
    wait_for_go_no_go()

    log("step 2: summarize go/no-go -> verdict")
    v = summarize(GONOGO_TAG, GONOGO_VERDICT)
    if not v:
        log("no verdict produced; stopping queue (inspect logs)")
        return 1
    if v.get("go"):
        log(f"VERDICT: GO (best acc {v.get('best_acc')} at plan "
            f"{v.get('best_plan')} {v.get('best_backbone')}, "
            f"ratio {v.get('best_ratio')}x)")
    else:
        log("VERDICT: NO-GO — no point at ratio>=2x held the R1 bar "
            "(<=5% drop). Per the pre-registered plan: STOP the compression "
            "track here; record the honest negative result; do NOT launch "
            "the full sweep. Next lane = calibrated-INT8 arm + residual "
            "fitting (N2-double-prime).")
        return 2

    log("step 3: full 12-point N2 sweep (seed 0) — this is a multi-hour run")
    rc = run_and_wait(bench_cmd(FULL_TAG), bench_env(None), FULL_LOG)
    if rc != 0:
        log(f"full sweep exited rc={rc}; check {FULL_LOG}")
        return rc

    log("step 4: summarize full sweep -> verdict")
    v2 = summarize(FULL_TAG, FULL_VERDICT)
    if v2:
        log(f"FULL SWEEP VERDICT: {'GO' if v2.get('go') else 'NO-GO'} "
            f"(best acc {v2.get('best_acc')} at "
            f"{v2.get('best_plan')}/{v2.get('best_backbone')}, "
            f"ratio {v2.get('best_ratio')}x)")

    log("=== queue complete ===")
    log("next lanes (manual, CPU-light): update technical-report.tex N2 rows "
        "from results/N2/summary.md; then calibrated-INT8 comparator arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
