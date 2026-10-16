"""Readers for the Movies dataset CSVs.

Everything is read as strings first, so the EDA can see exactly which values
fail to parse instead of pandas silently guessing types.
"""
import ast
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import DATA_DIR

# Columns in movies_metadata.csv stored as Python literals, e.g.
# "[{'id': 16, 'name': 'Animation'}]" (single quotes, so not valid JSON)
METADATA_NESTED: dict[str, type] = {
    "genres": list,
    "production_companies": list,
    "production_countries": list,
    "spoken_languages": list,
    "belongs_to_collection": dict,
}


def parse_literal(value: Any, expected: type | tuple[type, ...] = (list, dict)) -> Any:
    """Parse a Python-literal string such as "[{'id': 16, 'name': 'Animation'}]".

    Returns the parsed object (a list or dict), or None if the value is
    missing, unparseable, or not of the expected type.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        obj = ast.literal_eval(value)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return obj if isinstance(obj, expected) else None


def to_number(series: pd.Series) -> pd.Series:
    """Convert a string column to numbers; unparseable values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def read_csv_str(filename: str, data_dir: Path = DATA_DIR, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV with every column as string. Empty fields become NaN."""
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — is the dataset in {Path(data_dir)}?")
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], **kwargs)


def load_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return read_csv_str("movies_metadata.csv", data_dir)


def load_credits(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return read_csv_str("credits.csv", data_dir)


def load_keywords(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return read_csv_str("keywords.csv", data_dir)


def load_links(data_dir: Path = DATA_DIR, small: bool = False) -> pd.DataFrame:
    return read_csv_str("links_small.csv" if small else "links.csv", data_dir)


def iter_ratings(data_dir: Path = DATA_DIR, small: bool = False,
                 chunksize: int = 1_000_000) -> Iterator[pd.DataFrame]:
    """Yield ratings in chunks so the full 26M-row file never sits in memory at once.

    Each chunk has the columns userId (int32), movieId (int32), rating (float32)
    and timestamp (int64, Unix seconds).
    """
    path = Path(data_dir) / ("ratings_small.csv" if small else "ratings.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    yield from pd.read_csv(path, dtype=dtypes, chunksize=chunksize)
