"""Readers for the Movies dataset CSVs.

Everything is read as strings first, so the EDA can see exactly which values
fail to parse instead of pandas silently guessing types.
"""
import ast
from pathlib import Path

import pandas as pd

from .paths import DATA_DIR

# Columns in movies_metadata.csv stored as Python literals, e.g.
# "[{'id': 16, 'name': 'Animation'}]" (single quotes, so not valid JSON)
METADATA_NESTED = {
    "genres": list,
    "production_companies": list,
    "production_countries": list,
    "spoken_languages": list,
    "belongs_to_collection": dict,
}


def parse_literal(value, expected=(list, dict)):
    """Parse a Python-literal string. Returns None if it is missing,
    unparseable, or not of the expected type."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        obj = ast.literal_eval(value)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return obj if isinstance(obj, expected) else None


def to_number(series):
    """Convert a string column to numbers; unparseable values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def read_csv_str(filename, data_dir=DATA_DIR, **kwargs):
    """Read a CSV with every column as string. Empty fields become NaN."""
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — is the dataset in {Path(data_dir)}?")
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], **kwargs)


def load_metadata(data_dir=DATA_DIR):
    return read_csv_str("movies_metadata.csv", data_dir)


def load_credits(data_dir=DATA_DIR):
    return read_csv_str("credits.csv", data_dir)


def load_keywords(data_dir=DATA_DIR):
    return read_csv_str("keywords.csv", data_dir)


def load_links(data_dir=DATA_DIR, small=False):
    return read_csv_str("links_small.csv" if small else "links.csv", data_dir)


def iter_ratings(data_dir=DATA_DIR, small=False, chunksize=1_000_000):
    """Yield ratings in chunks so the full 26M-row file never sits in memory at once."""
    path = Path(data_dir) / ("ratings_small.csv" if small else "ratings.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    yield from pd.read_csv(path, dtype=dtypes, chunksize=chunksize)