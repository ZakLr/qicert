"""Shared helpers for bench modules. ASCII-only output (portable to any
terminal/CI encoding; the clean-env contract must never crash on stdout)."""

SMOKE_TABLES = frozenset()  # modules opt in via their own SMOKE set


def wants(rows: str, table: str, smoke: frozenset[str]) -> bool:
    """Row-selection contract: --rows=all | --rows=<table> | --rows=smoke."""
    if rows == "all":
        return True
    if rows == "smoke":
        return table in smoke
    return rows == table


def pending_row(exp_id: str, name: str, seeds: str, gpu_h: str) -> str:
    """A pre-registered experiment row that has not run yet (Phase 0)."""
    return f"| {exp_id} | {name} | {seeds} | {gpu_h} | [pending] no runs yet |"


def table_header(title: str, columns: list[str]) -> list[str]:
    out = [f"### {title}", ""]
    out.append("| " + " | ".join(columns) + " |")
    out.append("|" + "---|" * len(columns))
    return out
