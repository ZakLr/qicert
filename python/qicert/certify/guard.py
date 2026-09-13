"""B14h: runtime guard — load-time certificate enforcement + per-step action gate.

Load time (`check_certs_before_serve`):
  1. load manifest.json and RE-VERIFY its self-hash (a tampered manifest is
     refused, not served);
  2. verify every hashed artifact's sha256 against the manifest;
  3. refuse to serve if any certified bound is NaN/Inf (uncomputable bound =
     no certification = refuse).

Per step (`action_in_certified_set`):
  the certified claim is "the model's output stays within L·d of the
  reference action under input perturbation of diameter d".  The guard
  encodes this as an interval check on the action token; outside => the
  caller must take the safe fallback.  The interval logic itself is
  machine-validated against Z3 in tests/test_certify_guard.py (B14g).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from qicert.certify.manifest import hash_bytes, hash_file


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {"error": "no manifest found; cannot enforce guard"}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"manifest unreadable: {exc}"}


def _manifest_self_hash_ok(manifest: dict[str, Any]) -> bool:
    """Recompute the self-hash exactly as manifest.from_dir did."""
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    expected = hash_bytes(json.dumps(body, sort_keys=True).encode())
    return expected == manifest.get("manifest_sha256")


def check_certs_before_serve(run_dir: Path) -> dict[str, Any]:
    """Load-time gate. Returns {status: ok|refuse|warn, reason, ...}."""
    manifest = load_manifest(run_dir)
    if "error" in manifest:
        return {"status": "refuse", "reason": manifest["error"]}

    # 1. self-hash (tamper detection on the manifest itself)
    if not _manifest_self_hash_ok(manifest):
        return {"status": "refuse",
                "reason": "manifest self-hash mismatch — manifest was tampered"}

    artifacts: dict[str, str] = manifest.get("artifacts", {})
    cert_names = [n for n in artifacts
                  if n.startswith(("cert", "certificate", "lipschitz",
                                   "robustness"))]

    # 2. artifact integrity for every cert file listed
    for name in cert_names:
        path = run_dir / name
        if not path.exists():
            return {"status": "refuse",
                    "reason": f"certified artifact missing: {name}"}
        actual = hash_file(path)
        if actual != artifacts[name]:
            return {"status": "refuse",
                    "reason": f"artifact hash mismatch: {name}"}

    # 3. bound sanity: NaN/Inf certified bounds refuse service
    for name in cert_names:
        try:
            data = json.loads((run_dir / name).read_text())
        except Exception as exc:
            return {"status": "warn", "reason": f"could not parse {name}: {exc}"}
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if not isinstance(row, dict):
                continue
            L = row.get("lipschitz_float", row.get("lipschitz"))
            if L is not None:
                L = float(L)
                if math.isnan(L) or math.isinf(L):
                    return {"status": "refuse",
                            "reason": f"certified bound uncomputable ({name}): L={L}",
                            "cert_margin": L}
    if not cert_names:
        return {"status": "ok",
                "reason": "no certificate artifacts in this run"}
    return {"status": "ok",
            "reason": f"{len(cert_names)} certificate artifact(s) verified; "
                      f"bounds finite"}


class ActionDiversityMonitor:
    """Entropy gate over a sliding window of served action tokens.

    Motivation (measured 2026-09-13): deep TT compression collapses the VLA
    to a single modal action token. A per-layer Lipschitz certificate on such
    a model is sound-but-vacuous — the guard's ball check never fires because
    the model only ever emits one action. This monitor closes that hole by
    flagging degenerate output streams at deployment time.

    Design note: low diversity is a WARN, never a refuse. A robot
    holding position legitimately repeats one action token; refusing there
    would break correct behavior. The flag tells the operator (or the
    fallback controller) that the certificate currently attests to a
    degenerate policy.
    """

    def __init__(self, window: int = 64, min_entropy_bits: float = 1.0):
        if window < 2:
            raise ValueError("window must be >= 2")
        if min_entropy_bits < 0:
            raise ValueError("min_entropy_bits must be >= 0")
        self.window = int(window)
        self.min_entropy_bits = float(min_entropy_bits)
        self._hist: list[int] = []

    def update(self, action: int) -> dict[str, Any]:
        """Record one served action; warn on a full degenerate window."""
        self._hist.append(int(action))
        if len(self._hist) > self.window:
            self._hist.pop(0)
        if len(self._hist) < self.window:
            return {"status": "ok", "reason": "window filling",
                    "entropy_bits": None,
                    "n_unique": len(set(self._hist)),
                    "window": self.window}
        ent = self.entropy_bits()
        if ent < self.min_entropy_bits:
            return {"status": "warn",
                    "reason": "degenerate action stream: entropy "
                              f"{ent:.3f} bits < {self.min_entropy_bits:.3f} "
                              f"over last {self.window} actions",
                    "entropy_bits": ent,
                    "n_unique": len(set(self._hist)),
                    "window": self.window}
        return {"status": "ok", "reason": "action diversity sufficient",
                "entropy_bits": ent,
                "n_unique": len(set(self._hist)),
                "window": self.window}

    def entropy_bits(self) -> float:
        """Shannon entropy (bits) of the current window's empirical distribution."""
        n = len(self._hist)
        if n == 0:
            return 0.0
        counts: dict[int, int] = {}
        for a in self._hist:
            counts[a] = counts.get(a, 0) + 1
        return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))

    def reset(self) -> None:
        """Clear the window (e.g. on episode boundary)."""
        self._hist.clear()


def action_in_certified_set(action: int, reference: int, margin: float,
                            n_actions: int) -> bool:
    """Per-step gate: |action - reference| <= margin (token-line metric).

    This is the decision the report claims the deployment enforces.  It is
    intentionally trivial — the VALUE is that it is (a) enforced and logged,
    and (b) formally checked for soundness/completeness by Z3 (B14g).
    """
    if not math.isfinite(margin) or margin < 0:
        return False  # uncomputable/negative margin certifies nothing
    return abs(action - reference) <= margin and 0 <= action < n_actions
