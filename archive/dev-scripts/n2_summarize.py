"""Post-run N2 aggregator: turn recorded Pareto point dirs into the report table.

CPU-only, reads ONLY recorded artifacts (results/N2/<run_id>/run.json +
metrics.jsonl) — never re-derives a number.  Emits a markdown table grouped
by (backbone, plan) with ratio, eval acc, delta vs the FT reference, certified
sound layers, matched-batch flag, wall time — plus a run-level rollup
(points completed/failed/collapsed, best ratio within the 5% drop criterion).

Usage:
    PYTHONPATH=python python scripts/n2_summarize.py --exp-id N2 \
        [--handoff results/N1v2-ckpt] [--out results/N2/summary.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_points(n2_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(n2_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_json = run_dir / "run.json"
        if not run_json.exists():
            continue
        try:
            rec = json.loads(run_json.read_text())
        except Exception:
            continue
        results = rec.get("results", {})
        if "plan_fraction" not in results:
            continue  # not an N2 Pareto point
        rows.append({
            "run_id": rec.get("run_id", run_dir.name),
            "status": rec.get("status", "?"),
            "label": rec.get("label", ""),
            "backbone": results.get("backbone"),
            "plan": results.get("plan_fraction"),
            "rrank": results.get("residual_rank_cap"),
            "ratio": results.get("ratio"),
            "acc": results.get("eval_acc"),
            "delta": results.get("delta_vs_n1_ft"),
            "sound": results.get("cert_sound_layers"),
            "matched": results.get("matched_batch"),
            "wall": results.get("wall_sec"),
            "dir": run_dir.name,
            "command_line": rec.get("command_line", ""),
            "capture": rec.get("capture", ""),
            "start_utc": rec.get("start_utc", ""),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--exp-id", default="N2")
    ap.add_argument("--handoff", default="results/N1v2-ckpt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--invocation-tag", default=None,
                    help="only count points whose run.json command_line "
                         "contains this substring (e.g. N2-go-no-go-seed0). "
                         "Distinguishes a scored run from earlier smoke "
                         "points sitting in the same exp-id dir.")
    ap.add_argument("--verdict-json", default=None,
                    help="also write machine-readable go/no-go verdict JSON "
                         "(used by the sequential queue script). GO iff some "
                         "completed point at ratio >= 2x holds the R1 bar "
                         "(within 5%% drop of the FT reference).")
    args = ap.parse_args()

    n2_dir = Path(args.results_root) / args.exp_id
    if not n2_dir.exists():
        raise SystemExit(f"no {n2_dir}")

    # FT reference from the handoff baseline (matched-budget anchor).
    ft_ref = None
    baseline = Path(args.handoff) / "baseline_seed0.json"
    if baseline.exists():
        ft_ref = json.loads(baseline.read_text()).get("eval_acc_finetuned")

    rows = load_points(n2_dir)
    if args.invocation_tag:
        rows = [r for r in rows if args.invocation_tag in r["command_line"]]
    if not rows:
        raise SystemExit(f"no recorded N2 Pareto points under {n2_dir}")

    # Latest record per (backbone, plan[, residual rank]) by start_utc.
    latest: dict[tuple, dict] = {}
    for r in sorted(rows, key=lambda x: x["start_utc"]):
        key = (r["backbone"], r["plan"], r.get("rrank"))
        latest[key] = r  # later start_utc wins

    if ft_ref is None:
        ft_ref = 0.0  # collapse check disabled without the anchor
    lines = [
        f"# N2 results summary ({args.exp_id})",
        "",
        f"FT reference (seed 0, matched batch): {ft_ref if ft_ref is not None else 'n/a'}",
        f"Invocation filter: {args.invocation_tag or 'none (all recorded points)'}",
        "",
        "| Backbone | Plan | R' | Ratio | Eval acc | Delta vs FT | Cert sound | Matched | Wall s | Run |",
        "|----------|------|:--:|------:|---------:|------------:|-----------:|:-------:|-------:|-----|",
    ]
    completed = failed = collapsed = 0
    best_r = None
    for key in sorted(latest, key=lambda k: (str(k[0]), -(k[1] or 0),
                                             -(k[2] or 0))):
        bb, plan, rrank = key
        r = latest[key]
        if r["status"] != "completed":
            failed += 1
            lines.append(f"| {bb} | {plan} | - | FAILED ({r['status']}) | - | - | - | - | {r['run_id'][:8]} |")
            continue
        completed += 1
        acc = r["acc"] or 0.0
        dnum = r["delta"] if isinstance(r["delta"], (int, float)) and r["delta"] == r["delta"] else None
        if ft_ref is not None and dnum is not None and (ft_ref - acc) > 0.05:
            collapsed += 1
        dstr = f"{dnum:+.4f}" if dnum is not None else "n/a"
        if dnum is not None and (ft_ref - acc) <= 0.05 and r["ratio"]:
            if best_r is None or r["ratio"] > best_r:
                best_r = r["ratio"]
        lines.append(
            f"| {bb} | {plan} | {rrank if rrank else '-'} | {r['ratio']:.2f}x | {acc:.4f} | {dstr} | "
            f"{r['sound']} | "
            f"{'yes' if r['matched'] else 'NO'} | {r['wall']:.0f} | {r['run_id'][:8]} |")

    lines += [
        "",
        f"**Rollup:** {completed} completed, {failed} failed, {collapsed} "
        f"collapsed (>5% drop vs FT) — best ratio within the 5% criterion: "
        f"{best_r if best_r else 'none yet' }.",
        "",
        "*Numbers come only from recorded run.json artifacts; regenerate with "
        "`scripts/n2_summarize.py`.*",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\nwrote {args.out}")

    if args.verdict_json:
        # Pre-registered R1 bar: some completed point at ratio >= 2x whose
        # accuracy is within a 5% absolute drop of the FT reference.
        eligible = [r for r in latest.values()
                    if r["status"] == "completed" and r["ratio"]
                    and r["ratio"] >= 2.0
                    and isinstance(r["delta"], (int, float))
                    and r["delta"] == r["delta"]
                    and r["delta"] >= -0.05]
        best = max(eligible, key=lambda r: r["acc"] or 0.0) if eligible else None
        verdict = {
            "go": best is not None,
            "ft_ref": ft_ref,
            "bar": "delta >= -0.05 at ratio >= 2x (R1: <=5% drop at >=2x)",
            "best_acc": best["acc"] if best else None,
            "best_plan": best["plan"] if best else None,
            "best_rrank": best.get("rrank") if best else None,
            "best_backbone": best["backbone"] if best else None,
            "best_ratio": best["ratio"] if best else None,
            "points": [
                {"backbone": r["backbone"], "plan": r["plan"],
                 "rrank": r.get("rrank"), "ratio": r["ratio"],
                 "acc": r["acc"], "delta": r["delta"],
                 "sound": r["sound"], "status": r["status"]}
                for r in latest.values()
            ],
        }
        Path(args.verdict_json).write_text(json.dumps(verdict, indent=2))
        print(f"\nverdict: {'GO' if verdict['go'] else 'NO-GO'} "
              f"-> {args.verdict_json}")


if __name__ == "__main__":
    main()
