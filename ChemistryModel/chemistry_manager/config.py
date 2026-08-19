"""Small, explicit path configuration for the local manager."""

from pathlib import Path


DEFAULT_STATE_PATH = Path(".chemistry_manager") / "state.json"
DEFAULT_RUNS_ROOT = Path("runs")
DEFAULT_MOLECULE_ROOT = Path("molecules")
DEFAULT_TEACHER_ROOT = Path("teacher_data")
DEFAULT_TEACHER_REGISTRY_PATH = Path(".chemistry_manager") / "teacher_data.json"
