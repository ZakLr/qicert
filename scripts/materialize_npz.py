"""E8 - NPZ materializer. RUNS UNDER THE SIDE VENV (Python 3.12 + TF).

TensorFlow cannot install into the py3.14 project venv (no cp314 wheels),
so this ONE-OFF conversion job runs in a throwaway interpreter:
    py -3.12 -m venv %TEMP%\\venv-tf
    %TEMP%\\venv-tf\\Scripts\\pip install tensorflow-cpu tensorflow-datasets
    %TEMP%\\venv-tf\\Scripts\\python.exe scripts/materialize_npz.py

Reads the downloaded RLDS slice (weights/libero_spatial_no_noops) via
tensorflow_datasets and writes one NPZ per episode to
    weights/libero_spatial_no_noops_npz/ep_%04d.npz   keys:
        obs      (T, H, W, 3) uint8 frames (main camera)
        actions  (T, 7) float32
        language str scalar
plus the bridge-check reference values (per-dim action mean/std captured
from the tfds-read stream) to results/npz_bridge_check.json. After this
conversion TF is never needed again.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "weights" / "libero_spatial_no_noops"
# tfds.builder_from_directory wants the version dir (1.0.0/), not the suite root;
# older tfds versions accepted the root. Resolve to the version dir with the
# dataset_info.json so both layouts work.
if not (DATA_DIR / "dataset_info.json").exists():
    _verdirs = sorted(DATA_DIR.glob("*/dataset_info.json"))
    if _verdirs:
        DATA_DIR = _verdirs[0].parent
OUT_DIR = REPO / "weights" / "libero_spatial_no_noops_npz"
CHECK_JSON = REPO / "results" / "npz_bridge_check.json"


def main() -> int:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow_datasets as tfds

    assert DATA_DIR.exists(), f"missing RLDS slice: {DATA_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(CHECK_JSON.parent).mkdir(parents=True, exist_ok=True)

    builder = tfds.builder_from_directory(str(DATA_DIR))
    ds = builder.as_dataset(split="train")

    acts_sum = None
    acts_sqsum = None
    n_steps = 0
    n_eps = 0
    for idx, ep in enumerate(ds):
        steps = list(ep["steps"])
        frames = []
        actions = []
        language = None
        for st in steps:
            img = st["observation"]["image"].numpy()
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            frames.append(img)
            actions.append(st["action"].numpy().astype(np.float32))
            if language is None:
                _ld = st.get("language_instruction")
                if _ld is not None:
                    language = _ld.numpy().decode("utf-8") if hasattr(_ld, "numpy") else str(_ld)
        obs = np.stack(frames)
        act = np.stack(actions)
        np.savez_compressed(
            OUT_DIR / f"ep_{idx:04d}.npz",
            obs=obs, actions=act,
            language=np.array(language if language else "", dtype="<U256"))
        a = act.astype(np.float64)
        acts_sum = a.sum(axis=0) if acts_sum is None else acts_sum + a.sum(axis=0)
        acts_sqsum = ((a ** 2).sum(axis=0) if acts_sqsum is None
                      else acts_sqsum + (a ** 2).sum(axis=0))
        n_steps += len(steps)
        n_eps += 1
        if (idx + 1) % 20 == 0:
            print(f"  episode {idx + 1}: {n_steps} steps", flush=True)

    mean = acts_sum / n_steps
    var = acts_sqsum / n_steps - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    CHECK_JSON.write_text(json.dumps({
        "episodes": n_eps, "steps": n_steps,
        "action_dim": int(mean.shape[0]),
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "source": str(DATA_DIR),
        "note": "reference stats captured from the tfds stream during "
                "conversion; npz_loader --check must reproduce them <=1e-4",
    }, indent=2) + "\n")
    print(f"wrote {n_eps} episodes -> {OUT_DIR}")
    print(f"wrote {CHECK_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
