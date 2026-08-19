"""Small, explicit path configuration for the local manager."""

from pathlib import Path


DEFAULT_STATE_PATH = Path(".chemistry_manager") / "state.json"
DEFAULT_RUNS_ROOT = Path("runs")
DEFAULT_MOLECULE_ROOT = Path("molecules")

