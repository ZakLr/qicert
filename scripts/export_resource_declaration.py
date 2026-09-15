#!/usr/bin/env python3
"""Export the Resource Declaration (docs/resource-declaration.md).

The resource contract: every reported number to carry its resource context:
wall-clock, GPU-h, shots/circuit-evals (N/A here — this is the classical ML
side of the hybrid submission), seeds, and software stack. This script walks
results/ledger.csv + the per-run env.json/system.json artifacts and emits a
single markdown table a reviewer can diff against the claims in the report.

Usage:
    .venv312/Scripts/python.exe scripts/export_resource_declaration.py
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "results" / "ledger.csv"
OUT = REPO / "docs" / "resource-declaration.md"


def _load_run_extras(run_dir: Path) -> dict:
    """Pull torch/python versions + GPU name from a run dir's env/system json."""
    extras: dict = {}
    for name in ("env.json", "system.json"):
        p = run_dir / name
        if p.exists():
            try:
                extras.update(json.loads(p.read_text()))
            except Exception:
                pass
    return extras


def main() -> int:
    if not LEDGER.exists():
        print("no results/ledger.csv — nothing to declare", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(LEDGER.open(encoding="utf-8")))
    scored = [r for r in rows if r.get("status") == "completed"]
    failed = [r for r in rows if r.get("status") == "failed"]

    lines: list[str] = []
    lines.append("# Resource Declaration")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("Every claimed number must be traceable to a")
    lines.append("recorded run with its full resource cost. This table IS that")
    lines.append("trace, regenerated from `results/ledger.csv`.")
    lines.append("")
    lines.append("## Machine")
    lines.append("")
    lines.append(f"- Host: {platform.node()} ({platform.system()} {platform.release()})")
    lines.append(f"- CPU: {platform.processor() or 'n/a'}")
    gpu_names = sorted({r.get("gpu_name", "") for r in rows
                        if r.get("gpu_name") and r["gpu_name"] != "gpu_name"})
    for g in gpu_names:
        if g:
            lines.append(f"- GPU: {g}")
    lines.append("")
    lines.append("## Scored runs (status=completed)")
    lines.append("")
    lines.append("| experiment | run_id | seed | wall_sec | wall_min | energy_kwh | gpu | date |")
    lines.append("|---|---|---|---|---|---|---|---|")
    total_sec = 0.0
    for r in scored:
        try:
            wall = float(r.get("wall_sec") or 0)
        except ValueError:
            wall = 0.0
        total_sec += wall
        lines.append(
            f"| {r.get('experiment_id','')} | {r.get('run_id','')} "
            f"| {r.get('seed','')} | {wall:.1f} | {wall/60:.1f} "
            f"| {r.get('energy_kwh','')} | {r.get('gpu_name','')} "
            f"| {str(r.get('end_utc',''))[:10]} |")
    lines.append("")
    lines.append(f"**Total scored wall-clock: {total_sec/3600:.2f} GPU-h**")
    lines.append("")
    lines.append("## Failed / aborted runs (excluded from claims, kept for provenance)")
    lines.append("")
    if failed:
        lines.append("| experiment | run_id | seed | wall_sec | error (truncated) |")
        lines.append("|---|---|---|---|---|")
        for r in failed:
            err = (r.get("results") or "").replace("|", "/")[:100]
            lines.append(
                f"| {r.get('experiment_id','')} | {r.get('run_id','')} "
                f"| {r.get('seed','')} | {r.get('wall_sec','')} | {err} |")
    else:
        lines.append("_none_")
    lines.append("")
    lines.append("## Software stack (from latest completed run's env.json)")
    lines.append("")
    env_extras: dict = {}
    for r in reversed(scored):
        run_dir = REPO / "results" / r["experiment_id"] / \
            f"{r.get('run_id','')}_{r.get('run_tag','')}"
        if run_dir.exists():
            env_extras = _load_run_extras(run_dir)
            if env_extras:
                break
    for k in sorted(env_extras):
        v = str(env_extras[k])
        if len(v) > 120:
            v = v[:117] + "..."
        lines.append(f"- {k}: {v}")
    if not env_extras:
        lines.append("_env.json not found for any completed run_")
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(scored)} scored, {len(failed)} failed runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
