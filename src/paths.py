"""Project paths, resolved from this file so they work from any working directory."""
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
DATA_DIR: Path = PROJECT_ROOT / "data"      # put the dataset CSVs here
ENV_FILE: Path = PROJECT_ROOT / ".env"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"  # plots and other generated files
