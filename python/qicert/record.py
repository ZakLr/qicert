"""Run recorder — capture EVERYTHING so benchmarks never need re-running.

Decision 2026-08-16 (user): every benchmark/fine-tune writes complete artifacts:
``results/<EXP_ID>/<RUN_ID>/`` containing ``run.json`` (provenance + identity +
results), ``config.json`` (resolved, hashed), ``system.json`` (hardware),
``env.json`` (software pins), and ``metrics.jsonl`` (append-only trajectory).
A machine ledger ``results/ledger.csv`` ties every run to its fingerprint.

Guarantees:
  * stdlib + numpy only at import time — ``torch``, ``pynvml``, ``nvidia-smi``
    are probed lazily and degrade to ``None`` when absent (bench smoke / CI
    never import torch).
  * all artifacts are JSON / CSV, ASCII-safe, line-buffered, crash-tolerant
    (finalize is idempotent; a killed run still has its trajectory).
  * config_hash = sha256 of the canonical (sorted) config JSON — the ledger key.

Field inventory: docs/run-data-register.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from importlib import metadata
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sanitize(tag: str) -> str:
    """run tags are used in directory names: keep them small and safe."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", tag).strip("-")[:64]


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, numpy scalars converted, ASCII-safe."""
    def _conv(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        raise TypeError(f"not JSON-serializable: {type(o).__name__}")
    return json.dumps(obj, sort_keys=True, default=_conv, ensure_ascii=True)


def config_hash(config) -> str:
    """sha256 of the canonical config JSON — the run's fingerprint."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


def _try_import(name: str):
    try:
        mod = __import__(name)
        return mod
    except Exception:
        return None


# ---------------------------------------------------------------------------
# environment / system capture (best-effort, dependency-free)
# ---------------------------------------------------------------------------


def git_info(repo: str | None = None) -> dict:
    """qicert repo commit + dirty flag; repo defaults to this package's root."""
    base = repo or str(Path(__file__).resolve().parents[2])
    try:
        out = subprocess.run(
            ["git", "-C", base, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip() or None
        out = subprocess.run(
            ["git", "-C", base, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5)
        dirty = bool(out.stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty, "git_repo": base}
    except Exception:
        return {"git_commit": None, "git_dirty": None, "git_repo": base}


def _nvidia_smi(query: str) -> str | None:
    """Run nvidia-smi with a query; None when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def gpu_info() -> dict:
    """GPU snapshot: torch / pynvml / nvidia-smi, whichever are present."""
    info: dict = {}
    torch = _try_import("torch")
    if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
        try:
            dev = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            mem_total, mem_free = torch.cuda.mem_get_info(0)
            info.update({
                "gpu_name": dev,
                "gpu_compute_capability": ".".join(str(x) for x in cap),
                "vram_total_bytes": int(mem_total),
                "vram_free_bytes": int(mem_free),
                "gpu_count": torch.cuda.device_count(),
            })
        except Exception:
            pass

    nvml = _try_import("pynvml")
    if nvml is not None:
        try:
            nvml.nvmlInit()
            h = nvml.nvmlDeviceGetHandleByIndex(0)
            info["driver_version"] = nvml.nvmlSystemGetDriverVersion()
            try:
                info["gpu_power_w"] = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass
            try:
                info["gpu_temp_c"] = nvml.nvmlDeviceGetTemperature(
                    h, nvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
            try:
                info["gpu_util_pct"] = nvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                pass
        except Exception:
            pass

    smi = _nvidia_smi("name,driver_version,memory.total,power.draw,utilization.gpu,temperature.gpu")
    if smi:
        parts = [p.strip() for p in smi.split(",")]
        if len(parts) == 6 and not info.get("gpu_name"):
            info["gpu_name"] = parts[0]
        if not info.get("driver_version"):
            info["driver_version"] = parts[1]
        if not info.get("vram_total_bytes"):
            info["vram_total_bytes"] = int(parts[2].replace("MiB", "").strip()) * 1024 * 1024
        if not info.get("gpu_power_w"):
            try:
                info["gpu_power_w"] = float(parts[3])
            except ValueError:
                pass
        if not info.get("gpu_util_pct"):
            try:
                info["gpu_util_pct"] = int(parts[4])
            except ValueError:
                pass
        if not info.get("gpu_temp_c"):
            try:
                info["gpu_temp_c"] = float(parts[5])
            except ValueError:
                pass
    return info


def system_info() -> dict:
    """Host + container snapshot (best-effort; every probe is optional)."""
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "cpu_cores": os.cpu_count(),
        "hostname": platform.node(),
    }
    try:
        info["ram_total_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    try:
        usage = shutil.disk_usage(os.getcwd())
        info["disk_free_bytes"] = usage.free
    except Exception:
        pass
    # container probe
    info["in_container"] = (os.path.exists("/.dockerenv")
                            or os.path.exists("/run/.containerenv"))
    try:
        with open("/proc/1/cgroup", "r") as fh:
            info["container_cgroup"] = "docker" if "docker" in fh.read() else "none"
    except Exception:
        pass
    info["gpu"] = gpu_info()
    info["git"] = git_info()
    return info


def env_info() -> dict:
    """Software pins: versions of everything that matters (best-effort)."""
    names = [
        "torch", "transformers", "peft", "accelerate", "safetensors",
        "sentencepiece", "huggingface-hub", "numpy", "scipy", "pytest",
        "cudaq", "cuda-quantum", "bitsandbytes", "pynvml",
    ]
    pins: dict = {}
    for name in names:
        try:
            pins[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pins[name] = None
    # torch cuda build tag + arch list
    torch = _try_import("torch")
    if torch is not None:
        pins["torch_cuda_build"] = torch.__version__ or None
        try:
            pins["torch_cuda_arch_list"] = list(torch.cuda.get_arch_list())
        except Exception:
            pass
    # pip freeze dump (may be large — stored in env.json, not in memory long-term)
    try:
        out = subprocess.run([os.sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=30)
        pins["pip_freeze"] = out.stdout if out.returncode == 0 else None
    except Exception:
        pins["pip_freeze"] = None
    return pins


# ---------------------------------------------------------------------------
# the recorder
# ---------------------------------------------------------------------------


class RunRecorder:
    """Write complete, reproducible artifacts for one run.

    Usage::

        rec = RunRecorder(exp_id="N1", seed=0, config={...}, run_tag="minivla")
        rec.metric(step=0, loss=1.2, ...)      # trajectory, line-buffered
        rec.sample_power()                     # poll power/util, accumulate kWh
        rec.finalize(status="completed", results={...})
    """

    def __init__(self, exp_id: str, seed: int, config: dict, *,
                 out_root: str | os.PathLike = "results",
                 run_tag: str = "", capture: str = "default",
                 seed_spec: dict | None = None,
                 extra_identity: dict | None = None):
        self.exp_id = _sanitize(exp_id) or "run"
        self.seed = int(seed)
        self.run_id = uuid.uuid4().hex[:12]
        self.capture = capture if capture in ("default", "extra", "heavy") else "default"
        self.config = config
        self.config_hash = config_hash(config)
        tag = f"_{_sanitize(run_tag)}" if run_tag else ""
        self.run_dir = Path(out_root) / self.exp_id / f"{self.run_id}{tag}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._start_wall = time.time()
        self._start_mono = time.monotonic()
        self._energy_kwh = 0.0
        self._last_power_ts: float | None = None
        self._last_power_w: float | None = None
        self._finalized = False

        # identity + provenance (the run.json "running" state)
        ident = {
            "experiment_id": self.exp_id,
            "run_id": self.run_id,
            "run_tag": run_tag,
            "seed": self.seed,
            "seed_spec": seed_spec or {"torch": self.seed, "numpy": self.seed,
                                       "python": self.seed},
            "capture": self.capture,
            "config_hash": self.config_hash,
            "status": "running",
            "start_utc": _utc_now(),
            "command_line": _command_line(),
            "qicert_version": _pkg_version(),
        }
        if extra_identity:
            ident.update(extra_identity)

        self._write("run.json", ident)
        self._write("config.json", {"config": config})
        self._write("system.json", {"system": system_info()})
        self._write("env.json", {"env": env_info()})

        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._fh = open(self._metrics_path, "a", encoding="utf-8", errors="replace")
        self._fh.write(canonical_json({"event": "run_start",
                                       "t_wall": 0.0}) + "\n")
        self._fh.flush()

    # -- writing -----------------------------------------------------------

    def _write(self, name: str, payload: dict) -> None:
        (self.run_dir / name).write_text(
            canonical_json(payload) + "\n", encoding="utf-8", errors="replace")

    def metric(self, **fields) -> None:
        """Append one trajectory line (atomic-ish, line-buffered)."""
        line = {"t_wall": round(time.time() - self._start_wall, 4),
                "t_mono": round(time.monotonic() - self._start_mono, 4)}
        line.update({k: _py(v) for k, v in fields.items()})
        self._fh.write(canonical_json(line) + "\n")
        self._fh.flush()

    # -- power / energy ----------------------------------------------------

    def sample_power(self) -> dict:
        """Poll GPU power/util/temp; accumulate energy in kWh (power × dt)."""
        now = time.monotonic()
        gpu = gpu_info()
        p = gpu.get("gpu_power_w")
        sample = {"t_wall": round(time.time() - self._start_wall, 4),
                  "gpu_power_w": p,
                  "gpu_util_pct": gpu.get("gpu_util_pct"),
                  "gpu_temp_c": gpu.get("gpu_temp_c")}
        if p is not None and self._last_power_ts is not None and self._last_power_w is not None:
            dt = now - self._last_power_ts
            self._energy_kwh += self._last_power_w * dt / 3.6e6
        self._last_power_ts = now
        self._last_power_w = p
        sample["energy_kwh_cumulative"] = round(self._energy_kwh, 9)
        self.metric(**sample)
        return sample

    # -- finalize ----------------------------------------------------------

    def finalize(self, status: str = "completed", results: dict | None = None,
                 kill_criterion: str | None = None,
                 kill_verdict: str | None = None,
                 tolerance_note: str | None = None) -> dict:
        """Close the run: update run.json with results + write the ledger row.

        Idempotent: a second call re-writes run.json but never duplicates the
        ledger row (keyed on run_id).
        """
        if self._finalized:
            return self.summary()
        self._finalized = True
        wall_sec = time.time() - self._start_wall
        gpu = gpu_info()
        summary = {
            "experiment_id": self.exp_id,
            "run_id": self.run_id,
            "run_tag": self.run_dir.name,
            "seed": self.seed,
            "status": status,
            "config_hash": self.config_hash,
            "wall_sec": round(wall_sec, 2),
            "energy_kwh": round(self._energy_kwh, 9),
            "gpu_name": gpu.get("gpu_name"),
            "start_utc": (json.loads((self.run_dir / "run.json").read_text(
                encoding="utf-8")) or {}).get("start_utc"),
            "end_utc": _utc_now(),
            "kill_criterion": kill_criterion,
            "kill_verdict": kill_verdict,
            "tolerance_note": tolerance_note,
        }
        if results:
            summary["results"] = results
        ident = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        ident.update({"status": status, "end_utc": _utc_now(),
                      "wall_sec": round(wall_sec, 2),
                      "energy_kwh": round(self._energy_kwh, 9),
                      "results": results or {},
                      "kill_criterion": kill_criterion,
                      "kill_verdict": kill_verdict,
                      "tolerance_note": tolerance_note})
        self._write("run.json", ident)
        self._fh.close()
        self._append_ledger(summary)
        return summary

    def summary(self) -> dict:
        return json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))

    def _append_ledger(self, row: dict) -> None:
        ledger = Path(self.run_dir).parents[1] / "ledger.csv"
        cols = ["experiment_id", "run_id", "run_tag", "seed", "status",
                "config_hash", "wall_sec", "energy_kwh", "gpu_name",
                "start_utc", "end_utc", "kill_criterion", "kill_verdict",
                "results"]
        new = not ledger.exists()
        with open(ledger, "a", newline="", encoding="utf-8", errors="replace") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader()
            row = {k: _py(v) for k, v in row.items()}
            row["results"] = canonical_json(row.get("results") or {})
            w.writerow(row)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _py(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (Path, set, frozenset, tuple)):
        return str(v)
    return v


def _utc_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _command_line() -> str:
    try:
        return " ".join(_py(a) for a in __import__("sys").argv)
    except Exception:
        return ""


def _pkg_version() -> str | None:
    try:
        return metadata.version("qicert")
    except metadata.PackageNotFoundError:
        return None
