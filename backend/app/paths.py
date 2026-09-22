"""Shared data location, overridden before importing model-backed modules."""
import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("MECHLENS_DATA_DIR", Path(__file__).resolve().parents[1] / "data")).expanduser()
