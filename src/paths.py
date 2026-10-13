"""Project paths, resolved from this file so they work from any working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"  # put the dataset CSVs here
ENV_FILE = PROJECT_ROOT / ".env"
