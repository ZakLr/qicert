"""E8 - TF-free NPZ episode loader (main venv; never imports TensorFlow).

Serves episodes materialized by scripts/materialize_npz.py:
    NpzEpisodeDataset(root) -> len()==#episodes; item = (frames, actions,
    language) with frames (T, 224, 224, 3) uint8 (resized bicubic to the
    model resolution) and actions (T, 7) float32.

Design note: pixel NORMALIZATION is deliberately left to the consumer via
the fork's own ``vision_backbone.get_image_transform()`` (as
predict_action does) — duplicating normalization constants here would risk
a second convention drifting from the fork (repo convention).

Self-check (gate E8): ``python -m qicert.data.npz_loader --check`` recomputes
per-dimension action mean/std from the NPZ files and compares them against
the tfds-captured references in results/npz_bridge_check.json (<= 1e-4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "weights" / "libero_spatial_no_noops_npz"
DEFAULT_CHECK = REPO / "results" / "npz_bridge_check.json"
RESIZE = (224, 224)



class NpzEpisodeDataset:
    """Episode-level dataset over the materialized NPZ bridge."""

    def __init__(self, root: str | Path = DEFAULT_ROOT,
                 resize: tuple[int, int] | None = RESIZE):
        self.root = Path(root)
        self.files = sorted(self.root.glob("ep_*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"no ep_*.npz under {self.root} — run "
                f"scripts/materialize_npz.py first")
        self.resize = resize

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, str]:
        with np.load(self.files[i], allow_pickle=False) as z:
            obs = z["obs"]
            actions = z["actions"].astype(np.float32)
            language = str(z["language"])
        if self.resize is not None and obs.shape[1:3] != self.resize:
            from PIL import Image

            rs = np.stack([
                np.asarray(Image.fromarray(f).resize(
                    self.resize, Image.BICUBIC)) for f in obs])
            obs = rs.astype(np.uint8)
        return obs, actions, language


def _check() -> int:
    ds = NpzEpisodeDataset(resize=None)  # raw frames; stats only need actions
    acts = np.concatenate([ds[i][1].astype(np.float64) for i in range(len(ds))])
    mean, std = acts.mean(axis=0), acts.std(axis=0)
    ref = json.loads(Path(DEFAULT_CHECK).read_text())
    r_mean = np.array(ref["action_mean"])
    r_std = np.array(ref["action_std"])
    dm = float(np.abs(mean - r_mean).max())
    dsd = float(np.abs(std - r_std).max())
    print(f"episodes={len(ds)} steps={acts.shape[0]} "
          f"max|mean diff|={dm:.2e} max|std diff|={dsd:.2e}")
    ok = dm <= 1e-4 and dsd <= 1e-4
    print("BRIDGE CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_check() if "--check" in sys.argv else 0)
