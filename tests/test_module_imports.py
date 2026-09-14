"""Smoke test: every production module in src/ must import cleanly.

This doesn't test behavior — it only catches import-time breakage
(missing imports, circular imports, syntax errors that pass py_compile
but only surface on a real import, config validation failures) that
nothing else in CI exercises today, since the only other test file
only imports threshold_engine (and bot_config transitively).
"""
import importlib
import pathlib

import pytest

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src"
MODULE_NAMES = sorted(
    p.stem for p in SRC_DIR.glob("*.py")
    if p.stem != "__init__"
)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)
