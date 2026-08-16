"""Build the Kaggle kernel bundle for the N1 scored baseline.

Kaggle's `kaggle kernels push` uploads ONLY the code file + metadata — no
extra files. Since the qicert repo is private, the kernel must be
self-contained: this script embeds the qicert package (python/qicert,
bench/, scripts/patches/) as base64 inside the kernel script.

Usage (repo root):
    python scripts/build_kaggle_kernel.py --out kaggle/n1_baseline.py

Writes:
    kaggle/n1_baseline.py     the self-contained kernel script
    kaggle/kernel-metadata.json  push metadata (GPU T4, internet on)

Push + monitor + pull:
    kaggle kernels push -p kaggle
    kaggle kernels status zakilr/qicert-n1-baseline
    kaggle kernels output zakilr/qicert-n1-baseline -p results/kaggle
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Files included in the embedded bundle (relative to repo root).
# pyproject.toml is required for the kernel's `pip install -e` re-install.
BUNDLE = [
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "python/qicert",
    "bench",
    "scripts/patches/prismatic-transformers5.patch",
]

METADATA = {
    "id": "zakilr/qicert-n1-baseline",
    "title": "qicert N1 MiniVLA baseline (LoRA + INT8)",
    "code_file": "n1_baseline.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}


def make_tarball_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in BUNDLE:
            src = REPO / rel
            if src.is_dir():
                for f in sorted(src.rglob("*")):
                    if f.is_file() and "__pycache__" not in f.parts:
                        tf.add(f, arcname=str(f.relative_to(REPO)))
            else:
                tf.add(src, arcname=rel)
    return buf.getvalue()


def build(out: Path) -> None:
    tmpl = (Path(__file__).parent / "kaggle" / "kernel_template.py").read_text(
        encoding="utf-8")
    blob = base64.b64encode(make_tarball_bytes()).decode("ascii")
    assert "__QICERT_BUNDLE_B64__" in tmpl, "template marker missing"
    script = tmpl.replace("__QICERT_BUNDLE_B64__", blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")
    (out.parent / "kernel-metadata.json").write_text(
        json.dumps(METADATA, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(script)/1024:.0f} KiB) + kernel-metadata.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kaggle/n1_baseline.py")
    args = ap.parse_args()
    build(Path(args.out))
