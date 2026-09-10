"""transformers 5.x compatibility shims for the MiniVLA prismatic code.

The vendored prismatic tree (weights/code) imports private tokenizer module
paths that transformers 5.x removed (fast tokenizers now live at top level).
We register lightweight alias modules on sys.modules BEFORE prismatic imports
so `from transformers.models.qwen2.tokenization_qwen2_fast import
Qwen2TokenizerFast` resolves to the top-level class.

Called from bench workers right before `prismatic` is imported (see
bench/compress.py::_run_n1_seed). Idempotent; silent no-op on transformers
versions where the private path still exists.
"""
from __future__ import annotations

import importlib
import sys
import types

_DONE = False

# removed private path -> top-level attribute holding the class
_ALIASES = {
    "transformers.models.qwen2.tokenization_qwen2_fast": "Qwen2TokenizerFast",
}


def install() -> None:
    """Idempotently register alias modules for removed transformers paths."""
    global _DONE
    if _DONE:
        return
    for mod_name, attr in _ALIASES.items():
        if mod_name in sys.modules:
            continue
        try:
            importlib.import_module(mod_name)
            continue  # path still exists on this transformers version
        except ImportError:
            pass
        try:
            top = importlib.import_module("transformers")
            cls = getattr(top, attr)
        except (ImportError, AttributeError):
            continue
        mod = types.ModuleType(mod_name)
        mod.__dict__[attr] = cls
        sys.modules[mod_name] = mod
    _DONE = True
