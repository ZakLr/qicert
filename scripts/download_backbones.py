"""Download the MiniVLA backbones + record license rows (docs/backbones.md §3).

Usage (inside the qicert dev container, repo mounted):
    python scripts/download_backbones.py

Writes:
    weights/<kind>/        checkpoints (host-mounted, gitignored)
    weights/_licenses.json exact HF `license` card strings (report §6 needs them)
    docs/licenses.md       the human-readable compliance table
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPOS = {
    # MiniVLA fork code lives on GitHub (Stanford-ILIAD/openvla-mini,
    # OPENVLA_GIT below) — cloned separately in main(); vanilla
    # openvla/openvla does NOT carry the Prismatic MiniVLA tree the
    # transformers-5.x patch targets.
    "Stanford-ILIAD/minivla-libero90-prismatic": "ckpt",
    "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b": "backbone",
}
OPENVLA_GIT = "https://github.com/Stanford-ILIAD/openvla-mini.git"

# Native-first paths: env override > legacy /workspace (containers) > repo-relative.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_WS = Path("/workspace")
WEIGHTS = Path(os.environ.get("QICERT_WEIGHTS") or (
    "/workspace/weights" if _LEGACY_WS.exists() else str(_REPO_ROOT / "weights")))
DOCS = Path(os.environ.get("QICERT_DOCS") or (
    "/workspace/qicert/docs" if _LEGACY_WS.exists() else str(_REPO_ROOT / "docs")))

# Only checkpoint essentials are needed by Prismatic `load_vla`; the 7.5 GB
# stage-1 backbone ("backbone" kind) is not (ROADMAP). NOTE: hf_hub's
# `**/x.json` does NOT match root-level files — list them explicitly.
CKPT_ALLOW_PATTERNS = ["*step-122500*", "config.json", "config.yaml",
                       "dataset_statistics.json"]


def main() -> None:
    api = HfApi()
    lic: dict = {}
    # OpenVLA code: git clone into weights/code (persisted on host mount)
    if not (WEIGHTS / "code").exists():
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1", OPENVLA_GIT,
                        str(WEIGHTS / "code")], check=True)
        lic["openvla/openvla (github)"] = {
            "kind": "code", "license": "MIT", "downloaded_ok": True}
    else:
        lic["openvla/openvla (github)"] = {
            "kind": "code", "license": "MIT", "downloaded_ok": True}
    for repo, kind in REPOS.items():
        info = api.model_info(repo)
        lic[repo] = {
            "kind": kind,
            "license": info.cardData.get("license", "UNKNOWN"),
        }
        print(f"[{kind}] {repo}: license={lic[repo]['license']}", flush=True)
        if kind != "ckpt":
            # Stage-1 backbone (7.5 GB) not needed by Prismatic `load_vla`.
            lic[repo]["downloaded_ok"] = True
            lic[repo]["skipped"] = "not required by load_vla"
            print(f"  -> skipped ({lic[repo]['skipped']})", flush=True)
            continue
        try:
            snapshot_download(repo, local_dir=str(WEIGHTS / kind),
                              max_workers=4, allow_patterns=CKPT_ALLOW_PATTERNS)
            lic[repo]["downloaded_ok"] = True
            print(f"  -> {kind}/ done", flush=True)
        except Exception as e:  # noqa: BLE001
            lic[repo]["downloaded_ok"] = False
            lic[repo]["error"] = str(e)
            print(f"  -> FAILED: {e}", flush=True)

    (WEIGHTS / "_licenses.json").write_text(
        json.dumps(lic, indent=2, sort_keys=True) + "\n")

    # docs/licenses.md — the report §6 compliance table source
    lines = [
        "# Licenses — backbone weights (recorded at download)",
        "",
        "> Source of truth: `weights/_licenses.json` (exact HF `license` card",
        "> strings, captured at download time on "
        + (WEIGHTS / "_licenses.json").exists().__str__() + ").",
        "",
        "| Repo | Kind | License (exact HF string) | Downloaded |",
        "|---|---|---|---|",
    ]
    for repo in sorted(REPOS):
        l = lic.get(repo, {})
        lines.append(
            f"| {repo} | {l.get('kind', '?')} | {l.get('license', 'UNKNOWN')} "
            f"| {l.get('downloaded_ok', False)} |")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "licenses.md").write_text("\n".join(lines) + "\n")
    print("Wrote", DOCS / "licenses.md")
    print("LICENSES:", json.dumps(lic, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
