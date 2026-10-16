"""Cleaning: turn the CSVs into documents for MongoDB.

Pure data transformations - no database code here (see insert.py).
Every rule below comes from a finding in the EDA (src/eda/); the expected
numbers are from the full dataset and are asserted where possible.

Pipeline
--------
movies:  load -> repair broken rows -> drop duplicate ids -> types -> nulls
         -> parse nested -> drop unused columns
         + credits (one row per movie, cast grouped per person, crew trimmed)
         + keywords
         -> merge -> documents
users:   links (cleaned)  -> {movieId: tmdbId}
         movies           -> {tmdbId: genres}
         ratings (stream) -> one document per user

Status: functions marked TODO are skeletons to fill in. The rest is done.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

from .load import (iter_ratings, load_credits, load_keywords, load_links, load_metadata,
                   parse_literal, to_number)
from .paths import DATA_DIR
from .repair import repair_broken_rows

# --- Expected results on the full dataset (from the EDA) --------------------
EXPECTED_MOVIES = 45_433            # 45,466 rows - 3 second halves - 30 duplicate rows
EXPECTED_LINKS = 45_368             # links with tmdbId in metadata and matching imdb id
EXPECTED_RATINGS_FULL = 25_969_853  # ratings behind those links (full ratings.csv)

# --- Cleaning rules -----------------------------------------------------------
MONEY_MIN = 1000                    # budget/revenue below this -> null (mixed units, placeholders)
OVERVIEW_PLACEHOLDERS = {"No overview found.", "No Overview", "No movie overview available."}
TAGLINE_PLACEHOLDERS = {"-"}
INT_COLUMNS = ["budget", "revenue", "runtime", "vote_count"]
FLOAT_COLUMNS = ["vote_average", "popularity"]
BOOL_COLUMNS = ["adult", "video"]
DROP_COLUMNS = ["homepage", "poster_path"]
CAST_FIELDS = ["id", "name", "gender"]            # + order and characters, see group_cast
CREW_FIELDS = ["id", "name", "job", "department"]


# =============================================================================
# Movies
# =============================================================================

def clean_metadata(data_dir=DATA_DIR):
    """movies_metadata.csv -> one cleaned row per movie (45,433)."""
    raw = load_metadata(data_dir)
    df, repaired, dropped = repair_broken_rows(raw)
    assert to_number(df["id"]).notna().all(), "rows with a non-numeric id remain"
    assert df["title"].notna().all(), "rows without a title remain"
    print(f"metadata: {len(raw):,} rows, repaired {repaired} broken records, dropped {dropped}")

    df = drop_duplicate_ids(df)
    df = convert_types(df)
    df = apply_null_rules(df)
    df = parse_nested(df)
    return df.drop(columns=DROP_COLUMNS)


def drop_duplicate_ids(df):
    """TODO: id -> int, keep one row per id (the one with the highest vote_count).

    Hint: sort by the numeric vote_count descending, then drop_duplicates("id").
    Expected: 30 rows removed, 45,433 left.
    """
    raise NotImplementedError


def convert_types(df):
    """TODO: text -> real types.

    - INT_COLUMNS -> "Int64" (pandas integer that allows missing values)
    - FLOAT_COLUMNS -> float
    - BOOL_COLUMNS: "True"/"False" -> bool
    - release_date -> datetime (pd.to_datetime, format="%Y-%m-%d", errors="coerce")
    - text columns: strip whitespace, empty string -> missing
    """
    raise NotImplementedError


def apply_null_rules(df):
    """TODO: values that mean "unknown" -> missing.

    - budget, revenue: < MONEY_MIN -> missing, each column on its own
      (catches 0 as well). Before de-duplication that was 36,573 + 244 budgets
      and 38,055 + 154 revenues, so expect roughly 36,800 / 38,200 nulls.
    - runtime == 0 -> missing
    - vote_average -> missing where vote_count == 0
    - overview in OVERVIEW_PLACEHOLDERS, tagline in TAGLINE_PLACEHOLDERS -> missing
    original_language "xx" is kept on purpose (handled in Q9).
    """
    raise NotImplementedError


def parse_nested(df):
    """TODO: Python-literal strings -> lists/dicts.

    - genres, production_companies, production_countries, spoken_languages:
      parse_literal(v, list); missing -> [] . Keep the order of genres
      ([0] is the primary genre in Q5).
    - belongs_to_collection: parse_literal(v, dict) -> {"id", "name"} only,
      or None.
    """
    raise NotImplementedError


def clean_credits(data_dir=DATA_DIR):
    """TODO: credits.csv -> DataFrame with columns id, cast, crew (one row per movie).

    - id -> int, drop_duplicates("id") (44 rows; the 7 differing copies only
      differ in crew order)
    - cast: parse_literal(v, list) -> group_cast()
    - crew: parse_literal(v, list) -> trim_crew()
    """
    raise NotImplementedError


def group_cast(cast):
    """TODO: one entry per person, sorted by billing order.

    Input:  [{"id": 3, "name": .., "gender": 2, "order": 0, "character": "A", ...}, ...]
    Output: [{"id": 3, "name": .., "gender": 2, "order": 0, "characters": ["A", "B"]}, ...]

    - group by person id; order = the person's lowest order; characters in
      the order they appear, without duplicates
    - sort by (order, position in the original list) - 87 movies have ties
    Expected: 1,207 entries merged in 951 movies.
    """
    raise NotImplementedError


def trim_crew(crew):
    """TODO: keep only CREW_FIELDS for each crew entry. (No duplicates to remove.)"""
    raise NotImplementedError


def clean_keywords(data_dir=DATA_DIR):
    """TODO: keywords.csv -> DataFrame with columns id, keywords.

    - id -> int, drop exact duplicates (987 rows)
    - keywords: parse_literal(v, list) -> [{"id", "name"}]
    """
    raise NotImplementedError


def build_movies(meta, credits, keywords):
    """TODO: merge into one row per movie.

    meta.merge(credits, on="id", how="left", validate="one_to_one")
        .merge(keywords, on="id", how="left", validate="one_to_one")
    Missing cast/crew/keywords (e.g. the 1 movie without credits) -> [].
    """
    raise NotImplementedError


def to_movie_documents(movies):
    """One dict per movie, ready for insert_many. _id = TMDB id."""
    docs = []
    for row in movies.to_dict("records"):
        doc = {k: to_python(v) for k, v in row.items()}
        doc["_id"] = doc.pop("id")
        docs.append(doc)
    assert len(docs) == len({d["_id"] for d in docs}), "duplicate _id in movie documents"
    return docs


def clean_movies(data_dir=DATA_DIR):
    """Run the whole movies pipeline and return the documents."""
    meta = clean_metadata(data_dir)
    movies = build_movies(meta, clean_credits(data_dir), clean_keywords(data_dir))
    docs = to_movie_documents(movies)
    print(f"movies: {len(docs):,} documents (expected {EXPECTED_MOVIES:,})")
    return docs


# =============================================================================
# Users (ratings)
# =============================================================================

def build_link_map(movie_docs, data_dir=DATA_DIR):
    """TODO: links.csv -> {movieId: tmdbId}.

    Keep a link only if tmdbId is present, tmdbId is one of the movies, and
    the link's imdbId equals that movie's imdb_id (where the movie has one).
    These are the same rules as in src/eda/lost_ratings.py, so its
    classify_links() can be reused: it needs a DataFrame with columns "id"
    and "imdb_id", e.g. pd.DataFrame({"id": [d["_id"] for d in movie_docs],
    "imdb_id": [d["imdb_id"] for d in movie_docs]}), then keep reason == "ok".
    Expected: 45,368 links, every tmdbId used once (assert it).
    """
    raise NotImplementedError


def build_genre_lookup(movie_docs):
    """{tmdbId: [{"id", "name"}, ...]} - genres copied into every rating."""
    return {d["_id"]: d["genres"] for d in movie_docs}


def iter_user_documents(link_map, genre_lookup, data_dir=DATA_DIR, small=False, stats=None):
    """TODO: stream ratings.csv and yield one document per user.

    Document: {"_id": userId,
               "ratings": [{"movie_id": tmdbId, "rating": 4.5,
                            "timestamp": datetime (UTC), "genres": [{"id", "name"}]}, ...]}

    The file is sorted and grouped by userId (EDA step 6), so:
      for each chunk from iter_ratings(data_dir, small):
        - map movieId -> tmdbId with link_map; count and drop unmapped ratings
        - timestamp (Unix seconds) -> datetime: pd.to_datetime(..., unit="s", utc=True)
        - all users in the chunk except the last are complete -> yield them
        - the last user may continue in the next chunk -> carry their rows over
      after the last chunk, yield the carried user.
    Skip users with no mapped ratings.

    Count into `stats` (a Counter): ratings_read, ratings_unmapped,
    ratings_kept, users. Expected ratings_kept on the full file: 25,969,853.
    """
    raise NotImplementedError


# =============================================================================
# Helpers (done)
# =============================================================================

def to_python(value):
    """Convert pandas/numpy values to what pymongo can store.

    NaN/NaT/pd.NA -> None, numpy numbers -> int/float, Timestamp -> datetime,
    and the same recursively inside lists and dicts.
    """
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
        return None if isinstance(value, float) and math.isnan(value) else value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value
