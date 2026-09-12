#!/usr/bin/env python3
"""End-to-end runtime-guard demo (B14h narrative made concrete).

Builds a certified run directory (manifest + certificate artifacts from a
synthetic but REAL certificate chain: TT-SVD -> residual compensation ->
sound Lipschitz bound), then demonstrates the two gates the report claims:

  1. load-time gate  check_certs_before_serve(): manifest self-hash,
     artifact hashes, finite certified bounds
  2. per-step gate   action_in_certified_set(): the action token must lie
     inside the certified ball around the reference action

Scenarios demonstrated (all logged to the transcript):
  A. clean deployment        -> SERVE
  B. out-of-ball action      -> REFUSE (the headline: the machine says no)
  C. tampered certificate    -> REFUSE (hash mismatch)
  D. uncomputable bound (NaN)-> REFUSE
  E. missing manifest        -> REFUSE

Usage:
    PYTHONPATH=python python scripts/demo_guard.py [--out results/guard-demo]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from qicert.certify.guard import (  # noqa: E402
    action_in_certified_set,
    check_certs_before_serve,
)
from qicert.certify.manifest import from_dir, write_manifest  # noqa: E402
from qicert.compress_residual import (  # noqa: E402
    compressed_params,
    lipschitz_bound,
    svd_residual,
)
from qicert.kernels import get_backend  # noqa: E402


def _build_certified_run(run_dir: Path, tamper: bool = False,
                         nan_bound: bool = False) -> dict:
    """Compress a synthetic layer through the REAL certificate chain and
    write manifest.json + cert artifacts.  If `tamper`, the certificate is
    modified AFTER the manifest signs it — exactly the attack the load gate
    must catch (hash mismatch)."""
    k = get_backend()
    rng = np.random.default_rng(7)
    m_dims, n_dims = (16, 16), (16, 16)
    W = rng.standard_normal((256, 256))
    cs = k.tt_svd(W, m_dims, n_dims, (4,))
    comp = svd_residual(cs.arrays, m_dims, n_dims, W, residual_rank=8)
    L = lipschitz_bound(comp, m_dims, n_dims)
    if nan_bound:
        L = float("nan")

    run_dir.mkdir(parents=True, exist_ok=True)
    cert = {
        "layer": "demo.llm.layers.0.mlp.down_proj",
        "m_dims": list(m_dims), "n_dims": list(n_dims),
        "lipschitz": L,
        "residual_rank": comp["residual_rank"],
        "params": compressed_params(comp), "dense_params": int(W.size),
        "sound": bool(np.isfinite(L)),
    }
    cert_path = run_dir / "certificate-layer0.json"
    cert_path.write_text(json.dumps(cert, indent=2))
    manifest = from_dir(run_dir)
    write_manifest(run_dir, manifest)
    if tamper:
        # TAMPER AFTER SIGNING: lie about the bound post-manifest.
        cert["lipschitz"] = 0.001
        cert_path.write_text(json.dumps(cert, indent=2))
    return cert


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "guard-demo"))
    ap.add_argument("--margin", type=float, default=2.0,
                    help="certified action-token margin L*d (demo value)")
    ap.add_argument("--n-actions", type=int, default=64)
    args = ap.parse_args()

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    transcript: list[dict] = []

    def log(scenario: str, verdict: str, detail: str) -> None:
        entry = {"scenario": scenario, "verdict": verdict, "detail": detail}
        transcript.append(entry)
        print(f"  [{scenario:<28}] {verdict:<7} {detail}", flush=True)

    print("=== qicert runtime guard — end-to-end demo ===\n")

    # ---- Scenario A: clean deployment -----------------------------------
    runA = out_root / "A-clean"
    cert = _build_certified_run(runA)
    verdict = check_certs_before_serve(runA)
    log("A. load gate (clean)", verdict["status"].upper(), verdict["reason"])
    ok = action_in_certified_set(action=17, reference=16,
                                 margin=cert["lipschitz"] * 1.0,
                                 n_actions=args.n_actions)
    log("A. step gate (in-ball)", "SERVE" if ok else "REFUSE",
        f"action=17, reference=16, margin={cert['lipschitz'] * 1.0:.2f}")

    # ---- Scenario B: out-of-ball action ---------------------------------
    m = cert["lipschitz"] * 1.0
    bad = int(16 + m) + 3
    ok = action_in_certified_set(action=bad, reference=16, margin=m,
                                 n_actions=args.n_actions)
    log("B. step gate (OUT-of-ball)", "SERVE" if ok else "REFUSE",
        f"action={bad}, reference=16, margin={m:.2f} -> safe fallback engaged"
        if not ok else "unexpected")

    # ---- Scenario C: tampered certificate --------------------------------
    runC = out_root / "C-tampered"
    _build_certified_run(runC, tamper=True)
    verdict = check_certs_before_serve(runC)
    log("C. load gate (tampered)", verdict["status"].upper(), verdict["reason"])

    # ---- Scenario D: uncomputable bound ----------------------------------
    runD = out_root / "D-nan"
    _build_certified_run(runD, nan_bound=True)
    verdict = check_certs_before_serve(runD)
    log("D. load gate (NaN bound)", verdict["status"].upper(),
        verdict["reason"])

    # ---- Scenario E: missing manifest ------------------------------------
    runE = out_root / "E-missing"
    runE.mkdir(parents=True, exist_ok=True)
    verdict = check_certs_before_serve(runE)
    log("E. load gate (no manifest)", verdict["status"].upper(),
        verdict["reason"])

    served = sum(1 for t in transcript if t["verdict"] == "SERVE")
    refused = sum(1 for t in transcript if t["verdict"] == "REFUSE")
    print(f"\nSummary: {served} SERVE, {refused} REFUSE — the deployment "
          f"refuses every action/configuration it cannot certify.")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "transcript.json").write_text(json.dumps(
        {"transcript": transcript, "served": served, "refused": refused,
         "cert": cert}, indent=2))
    print(f"Transcript: {out_root / 'transcript.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
