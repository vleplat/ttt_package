"""Wrapper for the legacy `tt_lib.py`.

See `ttt_package.legacy.tucker2_lib` for rationale and usage notes.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_DIR = _ROOT / "legacy_backends"
_LEGACY_PATH = _LEGACY_DIR / "tt_lib.py"

if not _LEGACY_PATH.exists():
    raise ImportError(f"Legacy backend file not found at {_LEGACY_PATH}")

if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

try:
    globals().update(runpy.run_path(str(_LEGACY_PATH)))
except ModuleNotFoundError as e:
    raise ImportError(
        "The legacy backend `ttt_package.legacy.tt_lib` requires optional dependencies. "
        "Install them with: python -m pip install -e \".[legacy]\""
    ) from e

