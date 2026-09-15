#!/usr/bin/env python3
"""Generate every figure for technical-report-v2.tex from run artifacts.

Reproducibility contract: no figure may contain a number that
does not come from a file in results/.  This script reads:

  results/N1v2/*/metrics.jsonl                     -> Fig. training curve
  results/ledger.csv (N2 rows, 2026-09-12)         -> Fig. compression plane
  results/N2R2/*/run.json (weighted, full-split)   ->   same figure
  results/N8C/*/run.json (calibrated INT8)         ->   same figure
  results/N1v2-ckpt/baseline_seed0.json            ->   same figure (FT ref)
  results/safety/threearm_seed0.json               -> Fig. estimator race
  results/safety/conformal_seed0.json              -> Fig. conformal coverage
  results/E6-latency/*/run.json                    -> Fig. dense-vs-TT latency
  results/layer_spectra.csv                        -> Fig. spectra (Lipschitz inputs)
  results/E7-int8-ablation/*/run.json              -> Fig. INT8 calibration ablation

Output: docs/figures2/*.pdf (vector, 300-dpi-equivalent), printed inventory.
Run:    .venv312/Scripts/python.exe scripts/make_report_figures.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures2"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito palette (colorblind-safe)
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7F7F7F")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO)}")


# --------------------------------------------------------------------------
# Fig 1: fine-tuning loss curve (N1v2, seed 0, 1200 steps)
# --------------------------------------------------------------------------
def fig_training():
    best, best_n = None, 0
    for mj in glob.glob(str(REPO / "results" / "N1v2" / "*" / "metrics.jsonl")):
        n = sum(1 for line in open(mj, encoding="utf-8", errors="replace")
                if '"train_step"' in line)
        if n > best_n:
            best, best_n = mj, n
    assert best and best_n >= 1000, f"no usable N1v2 metrics.jsonl (best {best_n} steps)"
    steps, losses, accs = [], [], []
    for line in open(best, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "train_step":
            steps.append(r["step"]); losses.append(r["loss"]); accs.append(r.get("action_acc", np.nan))
    steps, losses, accs = map(np.asarray, (steps, losses, accs))
    o = np.argsort(steps)
    steps, losses, accs = steps[o], losses[o], accs[o]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(steps, losses, color=C_BLUE, lw=1.0, label="loss (CE)")
    # running median for trend visibility under batch noise
    w = 51
    if len(losses) > w:
        med = np.array([np.median(losses[max(0, i - w):i + 1]) for i in range(len(losses))])
        ax.plot(steps, med, color=C_RED, lw=1.6, label=f"running median (w={w})")
    ax.set_xlabel("fine-tune step"); ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    _save(fig, "fig_training.pdf")
    print(f"  source: {Path(best).parent.name} ({best_n} steps)")


# --------------------------------------------------------------------------
# Fig 2: compression-accuracy plane (the honest degradation story)
# --------------------------------------------------------------------------
def fig_compression_plane():
    pts = []

    # FT reference
    b = json.loads((REPO / "results" / "N1v2-ckpt" / "baseline_seed0.json").read_text())
    pts.append(("FT reference (no compression)", 1.0, b["eval_acc_finetuned"],
                C_GREY, "o", 60))

    # calibrated INT8 (N8C, full split)
    for rj in glob.glob(str(REPO / "results" / "N8C" / "*" / "run.json")):
        r = json.loads(Path(rj).read_text())
        s = r.get("results", {})
        if r.get("status") == "completed" and "full-split" in str(s.get("eval_scope", "")):
            pts.append(("calibrated INT8 (per-channel MSE)", s["ratio"], s["eval_acc"],
                        C_GREEN, "s", 55))

    # uniform TT (N2 ledger rows, 2026-09-12 batch-protocol screen)
    with open(REPO / "results" / "ledger.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["experiment_id"] != "N2" or not row["start_utc"].startswith("2026-09-12"):
                continue
            s = json.loads(row["results"])
            if s.get("backbone") == "TT" and row["status"] == "completed":
                pts.append(("uniform TT truncation (screen)", s["ratio"], s["eval_acc"],
                            C_RED, "v", 45))

    # weighted TT + residual repair (N2R2, full split)
    for rj in glob.glob(str(REPO / "results" / "N2R2" / "*" / "run.json")):
        r = json.loads(Path(rj).read_text())
        s = r.get("results", {})
        if (r.get("status") == "completed" and s.get("fit_mode") == "weighted"
                and "full-split" in str(s.get("eval_scope", ""))):
            seen = {(round(p[1], 3), round(p[2], 4)) for p in pts}
            if (round(s["ratio"], 3), round(s["eval_acc"], 4)) not in seen:
                pts.append(("TT + weighted residual repair", s["ratio"], s["eval_acc"],
                            C_BLUE, "D", 55))

    # N9C: repaired (LoRA) TT route, full split, re-derived dense certs —
    # the GO row.  Ratio is the honest figure (cores+residual+adapter bytes).
    for rj in glob.glob(str(REPO / "results" / "N9C" / "*" / "run.json")):
        r = json.loads(Path(rj).read_text())
        s = r.get("results", {})
        if (r.get("status") == "completed" and "full-split" in str(s.get("eval_scope", ""))
                and s.get("gate") == "GO"):
            pts.append(("TT + repair training (N9, GO)", s["ratio_honest"], s["eval_acc"],
                        C_ORANGE, "*", 140))

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    gate = b["eval_acc_finetuned"] - 0.05
    ax.axhspan(gate, 1.0, xmin=0, xmax=1, color=C_GREEN, alpha=0.07)
    ax.axhline(gate, color=C_GREEN, ls="--", lw=1.0)
    ax.axvline(2.0, color=C_GREY, ls=":", lw=1.0)
    ax.text(2.03, 0.045, "2$\\times$", color=C_GREY, fontsize=8)
    ax.text(3.15, gate + 0.012, "pre-registered bar (FT $-$ 0.05)", color=C_GREEN, fontsize=8)

    seen_labels = set()
    for label, r, a, c, m, sz in pts:
        ax.scatter(r, a, s=sz, c=c, marker=m, zorder=3,
                   label=label if label not in seen_labels else None)
        seen_labels.add(label)
    ax.set_xlabel("parameter compression ratio ($\\times$)")
    ax.set_ylabel("eval accuracy (full split)" )
    ax.set_xlim(0.5, 3.4); ax.set_ylim(-0.03, 0.55)
    ax.legend(loc="center right", fontsize=7.5)
    _save(fig, "fig_compression_plane.pdf")
    print(f"  points: {[(p[0], round(p[1],2), round(p[2],3)) for p in pts]}")


# --------------------------------------------------------------------------
# Fig 3: rare-event estimator race (N6)
# --------------------------------------------------------------------------
def fig_estimator_race():
    d = json.loads((REPO / "results" / "safety" / "threearm_seed0.json").read_text())
    rows = {r["arm"]: r for r in d["rows"]}
    p_true = d["p_true"]
    order = [a for a in ("naive-mc", "iqae-sim", "restart+gev") if a in rows]
    labels = {"naive-mc": "naive MC", "iqae-sim": "IQAE (amplification, sim.)",
              "restart+gev": "GEV tail fit"}
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for i, arm in enumerate(order):
        r = rows[arm]
        y = len(order) - 1 - i
        ax.errorbar([r["p_hat"]], [y],
                    xerr=[[r["p_hat"] - r["ci_lo"]], [r["ci_hi"] - r["p_hat"]]],
                    fmt="o", color=[C_RED, C_BLUE, C_ORANGE][i], capsize=3, ms=5)
        ax.text(r["ci_hi"] * 1.15, y, f"{int(r['n_queries']):.0e} queries"
                .replace("e+0", "e"), va="center", fontsize=8, color=C_GREY)
    ax.axvline(p_true, color="k", ls="--", lw=1.0)
    ax.text(p_true * 0.85, len(order) - 0.55, f"truth $p^*={p_true:.2e}$",
            fontsize=8, rotation=90, va="top")
    ax.set_xscale("log"); ax.set_xlabel("estimated failure probability $\\hat{p}$ (95% CI)")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[a] for a in reversed(order)])
    ax.set_ylim(-0.6, len(order) - 0.1)
    _save(fig, "fig_estimator_race.pdf")


# --------------------------------------------------------------------------
# Fig 4: conformal coverage vs target
# --------------------------------------------------------------------------
def fig_conformal():
    d = json.loads((REPO / "results" / "safety" / "conformal_seed0.json").read_text())
    rows = d["rows"]
    x = np.arange(len(rows))
    tgt = [r["target"] for r in rows]
    emp = [r["empirical"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(x, tgt, width=0.55, color="none", edgecolor=C_GREY, lw=1.0,
           label="target $1-\\alpha$")
    ax.bar(x, emp, width=0.35, color=C_BLUE, label="empirical coverage")
    for xi, t, e in zip(x, tgt, emp):
        ax.text(xi, min(t, e) - 0.045, f"{e:.3f}", ha="center", fontsize=8, color=C_BLUE)
    ax.set_xticks(x); ax.set_xticklabels([f"$\\alpha$={r['alpha']}" for r in rows])
    ax.set_ylabel("coverage on fresh test set")
    ax.set_ylim(0.80, 1.01)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig_conformal.pdf")


# --------------------------------------------------------------------------
# Fig 5: measured dense vs TT contraction latency (E6)
# --------------------------------------------------------------------------
def fig_latency():
    f = sorted(glob.glob(str(REPO / "results" / "E6-latency" / "*" / "run.json")))
    assert f, "no E6-latency run.json"
    rows = json.loads(Path(f[0]).read_text())["results"]["rows"]
    b16 = [r for r in rows if r["bond"] == 16]
    names = [r["matrix"] for r in b16]
    dense = [r["dense_ms"] for r in b16]
    tt = [r["tt_ms"] for r in b16]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(x - 0.18, dense, width=0.36, color=C_GREY, label="dense fp16")
    ax.bar(x + 0.18, tt, width=0.36, color=C_BLUE, label="TT contraction (bond 16)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("self_attn.", "").replace("mlp.", "mlp-") for n in names],
                       rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("per-layer time (ms, log)")
    ax.legend(fontsize=8)
    _save(fig, "fig_latency.pdf")
    ratio = [round(t / dd, 1) for t, dd in zip(tt, dense)]
    print(f"  TT/dense slowdowns at bond 16: {ratio}")


# --------------------------------------------------------------------------
# Fig 6: spectra of the FT layers (inputs the certificates are computed from)
# --------------------------------------------------------------------------
def fig_spectra():
    layers, pr, s1, types = [], [], [], []
    with open(REPO / "results" / "layer_spectra.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # proj column: "self_attn.q_proj" / "mlp.gate_proj" -> q/k/v/o/gate/up/down
            types.append(row["proj"].split(".")[-1].split("_")[0])
            pr.append(float(row["participation_ratio"])); s1.append(float(row["sigma1"]))
    pr, s1 = np.asarray(pr), np.asarray(s1)
    types = np.asarray(types)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3))
    ax = axes[0]
    for t, c in zip(("q", "k", "v", "o", "gate", "up", "down"),
                    (C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GREY, "#9E7BB5", "#56B4E9")):
        m = types == t
        if m.sum():
            ax.scatter(pr[m], s1[m], s=12, alpha=0.75, color=c, label=t)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("participation ratio (effective rank)")
    ax.set_ylabel("$\\sigma_1$ (spectral norm)")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    ax = axes[1]
    ax.hist(pr, bins=24, color=C_BLUE)
    ax.set_xlabel("participation ratio"); ax.set_ylabel("# layers")
    fig.tight_layout()
    _save(fig, "fig_spectra.pdf")
    print(f"  {len(pr)} layers; PR range {pr.min():.0f}-{pr.max():.0f}")


# --------------------------------------------------------------------------
# Fig 7: INT8 calibration ablation (E7)
# --------------------------------------------------------------------------
def fig_int8_ablation():
    f = sorted(glob.glob(str(REPO / "results" / "E7-int8-ablation" / "*" / "run.json")))
    assert f, "no E7 run.json"
    arms = json.loads(Path(f[0]).read_text())["results"]["arms"]
    names, drift = [], []
    for k in sorted(arms):
        if "perchannel" in k or "pertensor" in k:
            names.append(k); drift.append(arms[k]["action_rel_drift"])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(names))
    ax.bar(x, drift, width=0.5, color=[C_GREEN if "perchannel" in n else C_RED for n in names])
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("|", "\n") for n in names], fontsize=7.5)
    ax.set_ylabel("relative action drift")
    for xi, d in zip(x, drift):
        ax.text(xi, d + 0.02, f"{d:.3g}" if d else "0", ha="center", fontsize=8)
    _save(fig, "fig_int8_ablation.pdf")


if __name__ == "__main__":
    only = sys.argv[1:] or None
    jobs = {
        "training": fig_training, "plane": fig_compression_plane,
        "race": fig_estimator_race, "conformal": fig_conformal,
        "latency": fig_latency, "spectra": fig_spectra,
        "int8": fig_int8_ablation,
    }
    for name, fn in jobs.items():
        if only and name not in only:
            continue
        print(f"[{name}]")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    print("done ->", OUT)
