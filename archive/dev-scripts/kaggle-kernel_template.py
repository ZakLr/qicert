"""
N1 baseline + N2 compression sweep on Kaggle (MiniVLA-1B).

Self-contained kernel (Kaggle has no shell access; the qicert package is
embedded at build time by scripts/build_kaggle_kernel.py as QICERT_BUNDLE).

Runtime flow (all inside /kaggle/working):
  1. unpack embedded qicert bundle -> ./qicert-src  (python/ + bench/ + patches)
  2. clone the MiniVLA fork (Stanford-ILIAD/openvla-mini, public) -> ./code
  3. apply the bundled prismatic-transformers5.patch (Kaggle image ships
     transformers>=5, same as our container)
  4. install fork deps (peft, timm==0.9.10, draccus, dlimp via pip --no-deps).
     Kaggle's image already ships tensorflow 2.20 + tfds + jax + protobuf
     as a self-consistent stack; we do NOT touch it (installing an older
     tensorflow-cpu downgrades ml-dtypes and breaks the image's jax;
     pinning protobuf <6 breaks TF 2.20's tensorflow_metadata). We also
     uninstall the image's stale torchao 0.10.0 (peft >= 0.19 raises at
     LoRA-wrap time if torchao is present but < 0.16; we never use it).
  5. download from HF: final MiniVLA ckpt (5.2G) + LIBERO spatial slice (1.8G)
     + base Qwen2.5-0.5B / DINOv2 / SigLIP (pulled by the load path)
  6. warm the RLDS dataset-statistics cache (single pass, avoids races)
  7. run N1 (fine-tune + save-ckpt, INT8 SKIPPED — recorded in the scored
     N1 run): ONE SEED PER PROCESS, GPUs round-robin (2x Tesla T4 both get
     work). Each seed saves its fine-tuned backbone + the EXACT eval batch
     + per-seed baseline accuracy into FT_DIR as the N2 handoff.
  8. run N2 (6 bond plans x {TT, QTT} sweep): same process-per-seed GPU
     round-robin; each worker loads the VLA ONCE per seed and scores every
     compressed point on the matched N1 eval batch, with per-layer exact
     Lipschitz certificates.
  9. write results to /kaggle/working/results (Kaggle syncs this back)

Kernel metadata (kernel-metadata.json): enable_gpu, enable_internet,
machine_shape=NvidiaTeslaT4, kernel_type=script.

Config via env vars (Kaggle "Environment variables" panel; defaults = scored):
    QICERT_STEPS      train steps per seed   (default 2000)
    QICERT_BATCH      batch size             (default 4)
    QICERT_SEEDS      comma-separated seeds  (default 0,1,2)
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SRC = WORK / "qicert-src"
CODE = WORK / "code"
WEIGHTS = WORK / "weights"
RESULTS = WORK / "results"
# N1 -> N2 handoff: fine-tuned backbones + matched eval batches sidecars.
FT_DIR = WORK / "ft"
FORK_GIT = "https://github.com/Stanford-ILIAD/openvla-mini.git"

# Filled by build_kaggle_kernel.py (base64 of a tar.gz of python/ + bench/ +
# scripts/patches/). Kept as a separate marker so the template is greppable.
QICERT_BUNDLE_B64 = "__QICERT_BUNDLE_B64__"

# Scored config (pre-registered N1: 3 seeds, 8 GPU-h; plan's Kaggle budget).
SCORED_STEPS = int(os.environ.get("QICERT_STEPS", "2000"))
SCORED_BATCH = int(os.environ.get("QICERT_BATCH", "4"))
SCORED_SEEDS = [int(s) for s in os.environ.get("QICERT_SEEDS", "0,1,2").split(",") if s.strip()]


def _log(msg: str) -> None:
    print(f"[n1-kaggle] {msg}", flush=True)


def _sh(cmd: list[str], timeout: int | None = None) -> None:
    _log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, timeout=timeout)


def _pip(*pkgs: str, extra: list[str] | None = None) -> None:
    _sh([sys.executable, "-m", "pip", "install", "-q"] + list(pkgs) + (extra or []))


def _unpack_bundle() -> None:
    if SRC.exists():
        return
    blob = base64.b64decode(QICERT_BUNDLE_B64)
    tmp = WORK / "bundle.tar.gz"
    tmp.write_bytes(blob)
    with tarfile.open(tmp, "r:gz") as tf:
        tf.extractall(SRC)
    _log(f"bundle unpacked: {SRC}")


def _setup_paths() -> None:
    # The fork lives under /kaggle/working; put it on sys.path so
    # `import prismatic` resolves here.
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    os.environ.setdefault("PRISMATIC_DATA_ROOT", str(WORK / "prismatic-data"))
    os.environ.setdefault("HF_HOME", str(WORK / ".hf"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Point the bench module at this kernel's layout.
    os.environ["QICERT_CKPT"] = str(WEIGHTS / "ckpt" / "checkpoints" /
                                    "step-122500-epoch-55-loss=0.0743.pt")
    os.environ["QICERT_DATA_ROOT"] = str(WEIGHTS / "libero_spatial_no_noops")
    os.environ["QICERT_FORK"] = str(CODE)

    # Re-install qicert EDITABLE from the embedded bundle. The Kaggle image
    # has its own qicert (from this repo, baked) whose meta-path finder
    # shadows PYTHONPATH; re-pointing the editable install at our bundle
    # makes `import qicert` / `qicert.bench` resolve to the kernel's copy
    # (qicert.bench is a package-dir mapping to top-level bench/).
    _sh([sys.executable, "-m", "pip", "install", "-q", "-e", str(SRC)],
        timeout=300)


def _install_deps() -> None:
    # Kaggle image: torch + transformers>=5 + tensorflow 2.20 + tfds +
    # jax 0.7.2 + ml-dtypes + protobuf all preinstalled and self-consistent
    # (py3.12 OK). We add only the fork's pure-python deps + dlimp.
    #
    # CRITICAL: do NOT pip-install tensorflow-cpu, protobuf, or ml-dtypes
    # here. Installing tensorflow-cpu==2.18 downgrades ml-dtypes to <0.5,
    # breaking the image's jax 0.7.2 (TFLite util.py imports jax at module
    # load; jax raises ValueError on the version check, escaping the
    # try/except ImportError -> `import tensorflow` dies). And pinning
    # protobuf <6 breaks TF 2.20's tensorflow_metadata (needs >=6.31).
    # The image's stack is already consistent — leave it alone.
    _pip("peft", "timm==0.9.10", "draccus", "json-numpy", "wandb",
         extra=["--index-url", "https://pypi.org/simple"])
    # The image ships torchao 0.10.0 (old). peft >= 0.19 raises ImportError
    # at LoRA-wrap time when torchao is present but < 0.16. We never use
    # torchao (INT8 is torch native quantize_dynamic), so remove it — peft
    # then treats it as unavailable and skips cleanly.
    _sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    # dlimp pins TF 2.15 (py3.11-only): install --no-deps against the
    # image's TF 2.20.
    _sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "git+https://github.com/moojink/dlimp_openvla"])
    _log("deps installed")


def _clone_fork_and_patch() -> None:
    if not (CODE / "prismatic").exists():
        _sh(["git", "clone", "--depth", "1", FORK_GIT, str(CODE)])
    patch = SRC / "scripts" / "patches" / "prismatic-transformers5.patch"
    sentinel = CODE / "prismatic" / "models" / "backbones" / "llm" / ".qicert_patched"
    if not sentinel.exists():
        _sh(["git", "-C", str(CODE), "apply", str(patch)])
        sentinel.write_text("applied by qicert kaggle kernel\n")
    _log("fork patched")


def _download_weights() -> None:
    from huggingface_hub import snapshot_download
    ckpt = WEIGHTS / "ckpt" / "checkpoints" / "step-122500-epoch-55-loss=0.0743.pt"
    if not ckpt.exists():
        _log("downloading MiniVLA final ckpt (5.2G)...")
        snapshot_download(
            "Stanford-ILIAD/minivla-libero90-prismatic",
            local_dir=str(WEIGHTS / "ckpt"),
            allow_patterns=[
                "checkpoints/step-122500-epoch-55-loss=0.0743.pt",
                "config.json", "config.yaml", "dataset_statistics.json",
                "README.md",
            ],
            max_workers=4,
        )
    else:
        _log("ckpt present, skipping download")
    libero = WEIGHTS / "libero_spatial_no_noops"
    if not (libero / "libero_spatial_no_noops" / "1.0.0").exists():
        _log("downloading LIBERO spatial slice (1.8G)...")
        snapshot_download(
            "openvla/modified_libero_rlds",
            repo_type="dataset",
            local_dir=str(libero),
            allow_patterns=["libero_spatial_no_noops/**"],
            max_workers=4,
        )
    else:
        _log("LIBERO present, skipping download")


def _warm_dataset_stats() -> None:
    """Build the RLDS dataset once so the stats cache is written before the
    per-GPU workers race on it (same hash => same file, but avoid the race).

    The batch_transform only runs during iteration, so a dummy callable is
    enough to trigger construction-time stats computation — no model load."""
    try:
        _log("warming RLDS dataset-statistics cache (single pass)...")
        from prismatic.vla.datasets import RLDSDataset
        ds = RLDSDataset(
            Path(os.environ["QICERT_DATA_ROOT"]), "libero_spatial_no_noops",
            batch_transform=lambda x: x,   # never iterated here
            resize_resolution=(224, 224), shuffle_buffer_size=1000,
            image_aug=False,
        )
        _log(f"dataset ready: {ds.dataset_length} transitions; stats cached")
        del ds
        import gc
        gc.collect()
    except Exception as exc:  # never block the run on the warm pass
        _log(f"warm pass skipped ({type(exc).__name__}: {exc})")


def _n_gpus() -> int:
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:
        return 1


def _run_n1_multi_gpu() -> None:
    """One subprocess per seed, GPUs round-robin — both T4s get work.

    Each worker is the FULLY VALIDATED single-GPU bench path; we parallelize
    by splitting the seed list across GPUs instead of wrapping the model in
    DataParallel/DDP (no edge cases with dict pixel_values or peft-in-place).
    Falls back to sequential on 1-GPU sessions.
    """
    steps, batch, seeds = SCORED_STEPS, SCORED_BATCH, SCORED_SEEDS
    n_gpu = _n_gpus()
    _log(f"running N1 scored: steps/seed={steps} batch={batch} seeds={seeds} "
         f"gpus={n_gpu} (process-per-seed, round-robin)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    FT_DIR.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # subprocesses do NOT inherit sys.path — hand them the qicert package +
    # fork locations via PYTHONPATH (qicert alone resolves via the editable
    # install, but prismatic only exists on the parent's manual sys.path).
    pythonpath = os.pathsep.join([str(SRC / "python"), str(CODE)])
    env["PYTHONPATH"] = (pythonpath + os.pathsep + env["PYTHONPATH"]
                          if env.get("PYTHONPATH") else pythonpath)
    workers: list[subprocess.Popen] = []
    for i, seed in enumerate(seeds):
        gpu = i % n_gpu if n_gpu > 1 else 0
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, "-m", "qicert.bench.all",
               "--module", "compress", "--rows", "baseline-int8",
               "--out", str(RESULTS), "--exp-id", "N1",
               "--steps", str(steps), "--batch", str(batch),
               "--seeds", str(seed),
               "--save-ckpt", str(FT_DIR), "--skip-int8",
               "--run-tag", f"kaggle-t4-gpu{gpu}-seed{seed}"]
        _log("spawn: " + " ".join(str(c) for c in cmd))
        workers.append(subprocess.Popen(cmd, env=env))
        if (i + 1) % n_gpu == 0 and i + 1 < len(seeds):
            # fill the next GPU wave only after the current one finishes
            for w in workers:
                rc = w.wait(timeout=11 * 3600)
                if rc != 0:
                    _log(f"worker exited {rc}")
            workers = []

    rc = 0
    for w in workers:
        rc |= w.wait(timeout=11 * 3600)
    if rc:
        raise SystemExit(f"one or more N1 workers failed (rc={rc})")
    _log("all N1 workers done")


def _run_n2_multi_gpu() -> None:
    """N2 compression Pareto sweep: one subprocess per seed, GPUs round-robin.

    Consumes the fine-tuned backbones + matched eval batches N1 saved into
    FT_DIR (--save-ckpt sidecars). Each worker runs the full 6-plan x
    {TT, QTT} sweep for its seed, loading the VLA ONCE (36 model loads in
    the original design -> 3, one per seed). Falls back to sequential on a
    1-GPU session.
    """
    seeds = SCORED_SEEDS
    n_gpu = _n_gpus()
    _log(f"running N2 scored: plans=6 backbones=2 seeds={seeds} "
         f"gpus={n_gpu} (process-per-seed, round-robin)")
    RESULTS.mkdir(parents=True, exist_ok=True)

    missing = [s for s in seeds
               if not (FT_DIR / f"seed{s}.pt").exists()]
    if missing:
        raise SystemExit(f"N2: missing N1 fine-tuned ckpts for seeds {missing}")

    env = dict(os.environ)
    env["QICERT_FT_CKPT"] = str(FT_DIR)
    pythonpath = os.pathsep.join([str(SRC / "python"), str(CODE)])
    env["PYTHONPATH"] = (pythonpath + os.pathsep + env["PYTHONPATH"]
                          if env.get("PYTHONPATH") else pythonpath)
    workers: list[subprocess.Popen] = []
    for i, seed in enumerate(seeds):
        gpu = i % n_gpu if n_gpu > 1 else 0
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, "-m", "qicert.bench.all",
               "--module", "n2_sweep", "--rows", "compression-pareto",
               "--out", str(RESULTS), "--exp-id", "N2",
               "--seeds", str(seed),
               "--run-tag", f"kaggle-t4-gpu{gpu}-seed{seed}"]
        _log("spawn: " + " ".join(str(c) for c in cmd))
        workers.append(subprocess.Popen(cmd, env=env))
        if (i + 1) % n_gpu == 0 and i + 1 < len(seeds):
            for w in workers:
                rc = w.wait(timeout=11 * 3600)
                if rc != 0:
                    _log(f"N2 worker exited {rc}")
            workers = []

    rc = 0
    for w in workers:
        rc |= w.wait(timeout=11 * 3600)
    if rc:
        raise SystemExit(f"one or more N2 workers failed (rc={rc})")
    _log("all N2 workers done")


def main() -> None:
    _log(f"python {sys.version.split()[0]}; workdir {WORK}")
    t0 = time.time()
    _unpack_bundle()
    _setup_paths()
    _install_deps()
    _clone_fork_and_patch()
    _download_weights()
    _warm_dataset_stats()
    _run_n1_multi_gpu()
    _run_n2_multi_gpu()
    _log(f"done in {(time.time()-t0)/60:.1f} min; results in {RESULTS}")
    _log("files in /kaggle/working are synced back to the kernel output")


if __name__ == "__main__":
    main()
