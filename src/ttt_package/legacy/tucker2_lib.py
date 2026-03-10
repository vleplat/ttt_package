"""Wrapper for the legacy `tucker2_lib.py`.

To keep the main package lightweight, we ship the legacy audited backends as
standalone files at the repository root, and expose them under the
`ttt_package.legacy` namespace when working from a source checkout.

If you want to use these legacy backends, install the extra dependencies:

    python -m pip install -e ".[legacy]"
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_DIR = _ROOT / "legacy_backends"
_LEGACY_PATH = _LEGACY_DIR / "tucker2_lib.py"

if not _LEGACY_PATH.exists():
    raise ImportError(f"Legacy backend file not found at {_LEGACY_PATH}")

if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

try:
    globals().update(runpy.run_path(str(_LEGACY_PATH)))
except ModuleNotFoundError as e:
    raise ImportError(
        "The legacy backend `ttt_package.legacy.tucker2_lib` requires optional dependencies. "
        "Install them with: python -m pip install -e \".[legacy]\""
    ) from e


