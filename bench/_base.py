"""Shared helpers for bench modules. ASCII-only output (portable to any
terminal/CI encoding; the clean-env contract must never crash on stdout).

Recorder integration (Decision 2026-08-16): a module that runs a REAL
experiment MUST open a recorder via ``start_run`` and close it via
``finish_run`` — no recorder, no table (see docs/run-data-register.md).
``start_run`` returns None when the invocation did not pass ``--out``, so
modules stay runnable without capture (CI smoke).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchContext:
    """Per-invocation context threaded through bench modules.

    ``out`` is the results root (--out). When set, modules that run real
    experiments record complete artifacts (run.json/config.json/system.json/
    env.json/metrics.jsonl + ledger.csv) via qicert.record.RunRecorder.
    """

    out: str | None = None
    exp_id: str | None = None
    seed: int = 0
    run_tag: str = ""
    capture: str = "default"
    seed_spec: dict | None = None
    extra_identity: dict = field(default_factory=dict)
    steps: int | None = None       # N1: fine-tune steps (smoke vs scored)
    batch: int | None = None       # N1: batch size
    seeds: list[int] | None = None  # N1: explicit seed list
    steps_per_seed: int | None = None  # N1: steps per seed when seeds given

    @property
    def active(self) -> bool:
        return self.out is not None


def wants(rows: str, table: str, smoke: frozenset[str]) -> bool:
    """Row-selection contract: --rows=all | --rows=<table> | --rows=smoke."""
    if rows == "all":
        return True
    if rows == "smoke":
        return table in smoke
    return rows == table


def start_run(ctx: BenchContext | None, exp_id: str, label: str,
              config: dict) -> "RunRecorder | None":
    """Open a recorder for a real run; None when ctx is not capturing.

    ``label`` is a human tag (e.g. ``minivla-loRA-sliceA``); ``config`` is the
    resolved run config (it is hashed into config_hash).
    """
    if ctx is None or not ctx.active:
        return None
    from qicert.record import RunRecorder
    return RunRecorder(
        exp_id=exp_id, seed=ctx.seed, config=config,
        out_root=ctx.out, run_tag=f"{label}",
        capture=ctx.capture, seed_spec=ctx.seed_spec,
        extra_identity=ctx.extra_identity)


def finish_run(rec, status: str = "completed", results: dict | None = None,
               kill_criterion: str | None = None,
               kill_verdict: str | None = None,
               tolerance_note: str | None = None) -> None:
    """Close a recorder opened by start_run (no-op when rec is None)."""
    if rec is None:
        return
    rec.finalize(status=status, results=results,
                 kill_criterion=kill_criterion, kill_verdict=kill_verdict,
                 tolerance_note=tolerance_note)


def pending_row(exp_id: str, name: str, seeds: str, gpu_h: str) -> str:
    """A pre-registered experiment row that has not run yet (Phase 0)."""
    return f"| {exp_id} | {name} | {seeds} | {gpu_h} | [pending] no runs yet |"


def table_header(title: str, columns: list[str]) -> list[str]:
    out = [f"### {title}", ""]
    out.append("| " + " | ".join(columns) + " |")
    out.append("|" + "---|" * len(columns))
    return out
