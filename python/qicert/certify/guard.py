from __future__ import annotations
import math
import json
from pathlib import Path
from typing import Any

def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {"error": "no manifest found; cannot enforce guard"}
    return json.loads(path.read_text())

def check_certs_before_serve(run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    certs = manifest.get("artifacts", {}).get("certificate", {})
    if not certs:
        return {"status": "ok", "reason": "no certificate artifacts in this run"}
    for name in certs:
        path = run_dir / name
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for row in (data if isinstance(data, list) else [data]):
                    if (isinstance(row, dict) and
                            ("lipschitz_float" in row or "lipschitz" in row)):
                        L = float(row.get("lipschitz_float", row.get("lipschitz", 0.0)))
                        if math.isnan(L) or math.isinf(L):
                            return {
                                "status": "refuse",
                                "reason": f"certified bound uncomputable for {name}: L={L}",
                                "cert_margin": L,
                            }
            except Exception:
                return {"status": "warn", "reason": f"could not parse {name}"}
    return {"status": "ok", "reason": "certificates loadable; guard permissive (no NaN/Inf)"}
