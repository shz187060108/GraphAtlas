"""Make the source tree importable when scripts are executed directly.

This keeps `python scripts/<name>.py` working from a fresh checkout before an
editable install, while preserving the normal installed-package path.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
