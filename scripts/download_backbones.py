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
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPOS = {
    # OpenVLA code lives on GitHub (not HF) — clone separately in main().
    "Stanford-ILIAD/minivla-libero90-prismatic": "ckpt",
    "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b": "backbone",
}
OPENVLA_GIT = "https://github.com/openvla/openvla.git"

WEIGHTS = Path("/workspace/weights")
DOCS = Path("/workspace/qicert/docs")


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
        try:
            snapshot_download(repo, local_dir=str(WEIGHTS / kind), max_workers=4)
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
