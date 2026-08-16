"""Make project-root modules importable when a test is run directly."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRA_MODULE_DIRS = (
    PROJECT_ROOT,
    PROJECT_ROOT / "research" / "sapt",
    PROJECT_ROOT / "research" / "qm_residual",
    PROJECT_ROOT / "research" / "diagnostics",
    PROJECT_ROOT / "tools" / "profiling",
)

for module_dir in reversed(EXTRA_MODULE_DIRS):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
