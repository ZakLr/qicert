#!/usr/bin/env python3
"""Transformers 5.x compat shim for the Prismatic fork (qwen2 tokenizer aliases).

The fork's prismatic.vla.datasets + load.py import private tokenizer module
paths that were removed in transformers 5.x. Install aliases before any
prismatic import touches transformers, so load_vla / the RLDS pipeline can
import without ImportError.

No-op on older transformers where the paths still exist.

IMPORTANT — this file is a compatibility patch, not model code. Verifiability
requires treating it as a documented environment deviation in any claim that
depends on the upstream model path: we report the upstream commit plus this
patch's hash so a reviewer can reproduce the effective source.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def install(*, verbose: bool = False) -> None:
    """Alias the removed tokenizer-backed module paths for the Prismatic fork."""
    # The fork imports these specific private paths. If the real module already
    # exists (older transformers), this is a no-op and the alias below is
    # harmless.
    if _has_real("transformers.models.qwen2"):
        return

    for real_path, alias_path in _ALIASES:
        if _has_real(alias_path):
            continue
        _alias(real_path, alias_path, verbose=verbose)


def _has_real(path: str) -> bool:
    mod = sys.modules.get(path)
    if mod is not None:
        return True
    try:
        importlib.import_module(path)
        return True
    except ImportError:
        return False


def _alias(real_path: str, alias_path: str, *, verbose: bool = False) -> None:
    try:
        real = importlib.import_module(real_path)
    except Exception as exc:
        if verbose:
            print(f"[transformers5_compat] cannot alias {alias_path} -> "
                  f"{real_path}: {exc}", file=sys.stderr)
        return

    parent = _parent_module(alias_path)
    if parent is not None:
        setattr(parent, Path(alias_path).name, real)

    # Also register in sys.modules so downstream `import <alias>` finds it.
    sys.modules.setdefault(alias_path, real)
    if verbose:
        print(f"[transformers5_compat] aliased {alias_path} -> {real_path}",
              file=sys.stderr)


def _parent_module(path: str) -> ModuleType | None:
    parts = path.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        parent_path = ".".join(parts[:depth])
        mod = sys.modules.get(parent_path)
        if mod is not None:
            return mod
        try:
            mod = importlib.import_module(parent_path)
            return mod
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Removed private tokenizer module paths in transformers 5.x.
# Update this list if the fork starts importing new private paths.
# ---------------------------------------------------------------------------
_ALIASES: list[tuple[str, str]] = [
    # tokenizer + fast tokenizer backend modules the fork imports
    ("transformers.models.qwen2.tokenization_qwen2", "transformers.models.qwen2.tokenization_qwen2_fast"),
    ("transformers.models.qwen2.tokenization_qwen2_fast", "transformers.models.qwen2.tokenization_qwen2"),
]


# ---------------------------------------------------------------------------
# FlashAttention-2 → SDPA fallback (Windows hosts: flash_attn not installable)
# ---------------------------------------------------------------------------

_FLASH_NOTICE = (
    "[transformers5_compat] flash_attn not available on this host; "
    "falling back to SDPA attention (mathematically equivalent, fused kernel "
    "substituted). Logged as an environment deviation in run metadata."
)
_FLASH_FALLBACK_DONE = False


def _install_flash_fallback() -> bool:
    """Patch transformers' FA2 validator to degrade to SDPA when flash_attn is missing.

    The prismatic fork's qwen25 backbone defaults use_flash_attention_2=True, so any
    checkpoint load on a host without flash_attn dies with an ImportError inside
    transformers' validator. SDPA computes the same attention; this is a kernel
    substitution, not an algorithm change. Idempotent.
    """
    global _FLASH_FALLBACK_DONE
    if _FLASH_FALLBACK_DONE:
        return True
    try:
        import transformers.modeling_utils as mm
    except Exception as exc:  # pragma: no cover
        print(f"[transformers5_compat] flash fallback skipped: {exc}", file=sys.stderr)
        return False

    if not hasattr(mm, "PreTrainedModel") or not hasattr(
        mm.PreTrainedModel, "_check_and_enable_flash_attn_2"
    ):  # pragma: no cover
        print(
            "[transformers5_compat] transformers layout changed; flash fallback not applied",
            file=sys.stderr,
        )
        return False

    # Attribute access on the class returns the classmethod already bound to cls,
    # so the wrapper must NOT forward `cls` again (that shifts every positional
    # arg and collides with `torch_dtype` — the exact bug seen on first smoke).
    _orig = mm.PreTrainedModel.__dict__["_check_and_enable_flash_attn_2"].__func__

    def _sdpa_fallback(cls, config, *args, **kwargs):
        try:
            return _orig(cls, config, *args, **kwargs)
        except (ImportError, ValueError) as exc:
            print(_FLASH_NOTICE, file=sys.stderr)
            print(
                f"[transformers5_compat] FA2 unavailable ({exc.__class__.__name__}); using sdpa",
                file=sys.stderr,
            )
            config._attn_implementation = "sdpa"
            return config

    _sdpa_fallback.__doc__ = "FA2→SDPA fallback (see qicert.transformers5_compat)"

    mm.PreTrainedModel._check_and_enable_flash_attn_2_orig = _orig  # kept for audit
    mm.PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(_sdpa_fallback)
    _FLASH_FALLBACK_DONE = True
    return True


def install_all() -> None:
    """Install ALL compat shims (tokenizer aliases + FA2→SDPA fallback). Call once, before prismatic imports."""
    install()
    _install_flash_fallback()
    _install_peft_torchao_compat()


# Back-compat: existing call sites `install()` must get the full shim set.
_orig_install = install


def install(*, verbose: bool = False) -> None:  # noqa: F811
    _orig_install(verbose=verbose)
    _install_flash_fallback()
    _install_peft_torchao_compat()


# ---------------------------------------------------------------------------
# peft 0.14 ↔ torchao 0.18 skew: LinearActivationQuantizedTensor marker stub
# ---------------------------------------------------------------------------

def _install_peft_torchao_compat() -> bool:
    """Unblock peft's LoRA dispatcher under torchao 0.18.

    peft 0.14's dispatch_torchao imports
    torchao.quantization.LinearActivationQuantizedTensor unconditionally whenever
    torchao is importable, but torchao 0.18 removed that class. peft only uses the
    name for an isinstance() check against weights already quantized by torchao;
    our LoRA targets are plain fp16/fp32 Linears, so a marker stub makes the import
    succeed and the branch correctly False. Version-guarded + idempotent.
    """
    try:
        import torchao.quantization as tq
    except Exception:
        return True  # torchao absent → peft never imports the name; nothing to do
    if hasattr(tq, "LinearActivationQuantizedTensor"):
        return True
    try:
        from torchao.quantization import LinearActivationQuantizedTensor  # noqa: F401
        return True  # importable after all (older torchao); no stub needed
    except ImportError:
        pass

    class LinearActivationQuantizedTensor:  # pragma: no cover — marker only
        """Compat marker for peft 0.14 ↔ torchao 0.18 skew (never instantiated here)."""

    tq.LinearActivationQuantizedTensor = LinearActivationQuantizedTensor
    print(
        "[transformers5_compat] injected LinearActivationQuantizedTensor marker "
        "(peft 0.14 expects it; torchao 0.18 removed it)",
        file=sys.stderr,
    )
    return True


if __name__ == "__main__":
    install_all()
    import transformers

    print("transformers", transformers.__version__)
