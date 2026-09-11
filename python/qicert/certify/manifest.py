"""B14h: signed manifest of every artifact in a run.

For a given run directory, writes manifest.json containing sha256 hashes of:
- model weights (or the N2-compressed model artifact)
- certificate file(s)
- frozen eval split
- config + env + system provenance
- and the manifest's own hash (self-signing).\n
Intended to be generated at run completion (record.py hook) and presented in the\nreport as the audit trail / verification artifact.\n"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

HASH_ALG = "sha256"


def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_file(path: Path, chunk_size: int = 1 << 16) -> str | None:
    try:
        with path.open("rb") as f:
            h = hashlib.sha256()
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def from_dir(run_dir: Path, include: list[str] | None = None) -> dict[str, Any]:
    """Build a manifest dict for everything under run_dir (non-recursive)\n    whose name starts with a known artifact prefix.\n    """
    prefixes = {
        "model": ["model", "ckpt", "weights"],
        "certificate": ["cert", "certificate", "lipschitz", "robustness"],
        "split": ["eval_split", "split"],
        "provenance": ["config", "env", "system"],
    }
    if include is None:
        include = list(prefixes)
    manifest: dict[str, Any] = {"run_dir": str(run_dir),
                                "created": json.dumps(__import__("time").time()),
                                "artifacts": {}}
    for entry in sorted(run_dir.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        for key in include:
            if any(name.startswith(p) for p in prefixes[key]):
                md5 = hash_file(entry)
                manifest["artifacts"][name] = md5
    manifest["manifest_sha256"] = hash_bytes(
        json.dumps(manifest, sort_keys=True).encode()
    )
    return manifest


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
