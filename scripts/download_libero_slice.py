"""Download a LIBERO RLDS slice for N1 fine-tuning (openvla/modified_libero_rlds).

Suite chosen: libero_spatial_no_noops (1.91 GB) — one of the four suites
MiniVLA-1B was fine-tuned on; a *slice* per N1's pre-registration
(submission/08-experiments.md: "fine-tune on a LIBERO slice"). License:
MIT (HF card, recorded here + docs/licenses.md).

Usage (container, weights mounted):
    python scripts/download_libero_slice.py

Writes:
    weights/libero_spatial_no_noops/   RLDS/TFDS dataset dir
    docs/licenses.md                   appended license row
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SUITE = "libero_spatial_no_noops"
REPO = f"openvla/modified_libero_rlds"
# Native-first paths: env override > legacy /workspace (containers) > repo-relative.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_WS = Path("/workspace")
WEIGHTS = Path(os.environ.get("QICERT_WEIGHTS") or (
    "/workspace/weights" if _LEGACY_WS.exists() else str(_REPO_ROOT / "weights")))
DOCS = Path(os.environ.get("QICERT_DOCS") or (
    "/workspace/qicert/docs" if _LEGACY_WS.exists() else str(_REPO_ROOT / "docs")))


def main() -> int:
    api = HfApi()
    info = api.dataset_info(REPO)
    lic = (info.cardData or {}).get("license", "UNKNOWN")
    print(f"license card: {lic}")
    # Download into WEIGHTS root so files land canonically at
    # WEIGHTS/<SUITE>/1.0.0/ (RLDS layout RLDSDataset(data_dir=WEIGHTS) expects);
    # a local_dir of WEIGHTS/<SUITE> would nest the suite one level deeper.
    snapshot_download(REPO, repo_type="dataset", allow_patterns=f"{SUITE}/*",
                      local_dir=str(WEIGHTS))
    # record
    lic_file = WEIGHTS / "_libero_licenses.json"
    rec = {REPO: {"suite": SUITE, "license": lic, "note": "N1 slice"}}
    if lic_file.exists():
        rec = {**json.loads(lic_file.read_text()), **rec}
    lic_file.write_text(json.dumps(rec, indent=2) + "\n")
    print("Wrote", lic_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
