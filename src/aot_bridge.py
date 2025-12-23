"""
Import bridge: prefer AOT-compiled helpers; fallback to JIT transparently.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

# ------------------------------------------------------------------
# 1.  Locate / load the shared object (if it exists)
# ------------------------------------------------------------------
_AOT_MODULE: Any = None
_AOT_AVAILABLE = False

def _try_load_aot() -> bool:
    global _AOT_MODULE, _AOT_AVAILABLE

    build_dir = Path(__file__).resolve().parent.parent / "build" / "aot"
    candidate = build_dir / "aot_compiled.so"          # Linux
    if not candidate.exists():
        candidate = build_dir / "aot_compiled.pyd"     # Windows
    if not candidate.exists():
        return False

    try:
        # Temporarily inject build dir into sys.path
        sys.path.insert(0, str(build_dir))
        import aot_compiled  # type: ignore[import]
        _AOT_MODULE = aot_compiled
        _AOT_AVAILABLE = True
        sys.path.remove(str(build_dir))
        return True
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"AOT module not loadable ({exc}) – falling back to JIT", stacklevel=2)
        return False

_AOT_AVAILABLE = _try_load_aot()

# ------------------------------------------------------------------
# 2.  Public façade
# ------------------------------------------------------------------
def get_impl(func_name: str) -> Callable[..., Any]:
    """
    Return the *fastest* implementation available for <func_name>.
    AOT is preferred; otherwise the JIT-decorated version from macd_unified.
    """
    if _AOT_AVAILABLE and hasattr(_AOT_MODULE, func_name):
        return getattr(_AOT_MODULE, func_name)

    # fallback – import JIT version
    from . import macd_unified as mu  # local import to avoid circularity
    return getattr(mu, func_name)
