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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .load import (iter_ratings, load_credits, load_keywords, load_links, load_metadata,
                   parse_literal, to_number)
from .paths import DATA_DIR
from .repair import repair_broken_rows
from .schema import (CastMember, CrewMember, Genre, MovieDoc, RawCastEntry, RawCrewEntry,
                     UserDoc)

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
TEXT_COLUMNS = ["imdb_id", "title", "original_title", "original_language", "overview",
                "tagline", "status"]
DROP_COLUMNS = ["homepage", "poster_path"]
CAST_FIELDS = ["id", "name", "gender"]            # + order and characters, see group_cast
CREW_FIELDS = ["id", "name", "job", "department"]


# =============================================================================
# Movies
# =============================================================================

def clean_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
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


def drop_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """id -> int, keep one row per id (the one with the highest vote_count).

    vote_count is still text here, so it is converted for sorting - sorting
    the text would put "9" above "150". kind="stable" keeps ties (most
    duplicates have equal vote_count) in file order, so reruns give the same
    result. Expected: 30 rows removed, 45,433 left.
    """
    df = df.copy()
    df["id"] = to_number(df["id"]).astype(int)
    before = len(df)
    df = (df.assign(_votes=to_number(df["vote_count"]))
            .sort_values("_votes", ascending=False, kind="stable", na_position="last")
            .drop_duplicates("id", keep="first")
            .drop(columns="_votes")
            .sort_index()
            .reset_index(drop=True))
    print(f"duplicate ids: removed {before - len(df)} rows, {len(df):,} movies left")
    assert df["id"].is_unique, "duplicate ids remain"
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Text -> real types. Only changes types; deciding what counts as
    "unknown" is apply_null_rules' job.

    - INT_COLUMNS -> "Int64" (capital I: the pandas integer that allows
      missing values; plain int would crash on the first NaN)
    - FLOAT_COLUMNS -> float (vote_average has decimals, e.g. 6.5)
    - BOOL_COLUMNS: "True"/"False" -> nullable boolean (missing stays missing)
    - release_date -> datetime; unparseable or missing -> NaT
    - text columns: strip whitespace; empty or whitespace-only -> missing
    The nested columns (genres, ...) stay text until parse_nested.
    """
    df = df.copy()

    for col in INT_COLUMNS:
        # .round() first: "104.0" is a whole number, but astype("Int64")
        # refuses anything with decimals instead of silently cutting them off
        df[col] = to_number(df[col]).round().astype("Int64")
    for col in FLOAT_COLUMNS:
        df[col] = to_number(df[col]).astype(float)
    for col in BOOL_COLUMNS:
        df[col] = df[col].map({"True": True, "False": False}).astype("boolean")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", format="%Y-%m-%d")

    blanked = 0
    for col in TEXT_COLUMNS:
        stripped = df[col].str.strip()
        is_blank = stripped == ""
        blanked += int(is_blank.sum())
        df[col] = stripped.mask(is_blank)
    print(f"types converted; {blanked} blank text values -> missing")
    return df


def apply_null_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Values that mean "unknown" -> missing. Prints how many values each rule
    changed (numbers for the report).

    - budget, revenue: < MONEY_MIN -> missing, each column on its own
      (catches 0 as well). Before de-duplication that was 36,573 + 244 budgets
      and 38,055 + 154 revenues, so expect roughly 36,800 / 38,200.
    - runtime == 0 -> missing
    - vote_average -> missing where vote_count == 0
    - overview in OVERVIEW_PLACEHOLDERS, tagline in TAGLINE_PLACEHOLDERS -> missing
    original_language "xx" is kept on purpose (handled in Q9).
    """
    df = df.copy()
    changed: dict[str, int] = {}

    def to_missing(col: str, condition: pd.Series, rule: str) -> None:
        """Set df[col] to missing where condition is True, and count the changes."""
        condition = condition.fillna(False).astype(bool)   # a missing comparison = no change
        changed[rule] = int((condition & df[col].notna()).sum())
        df[col] = df[col].mask(condition)

    for col in ("budget", "revenue"):
        to_missing(col, df[col] < MONEY_MIN, f"{col} < {MONEY_MIN:,} (incl. 0)")
    to_missing("runtime", df["runtime"] == 0, "runtime == 0")
    to_missing("vote_average", df["vote_count"] == 0, "vote_average where vote_count == 0")
    to_missing("overview", df["overview"].isin(OVERVIEW_PLACEHOLDERS), "overview placeholder")
    to_missing("tagline", df["tagline"].isin(TAGLINE_PLACEHOLDERS), "tagline placeholder")

    print("null rules:")
    for rule, n in changed.items():
        print(f"  {rule:<38} {n:>7,} values -> missing")
    return df


def parse_nested(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: Python-literal strings -> lists/dicts.

    - genres, production_companies, production_countries, spoken_languages:
      parse_literal(v, list); missing -> [] . Keep the order of genres
      ([0] is the primary genre in Q5).
    - belongs_to_collection: parse_literal(v, dict) -> {"id", "name"} only,
      or None.
    """
    raise NotImplementedError


def clean_credits(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """credits.csv -> DataFrame with columns id, cast, crew (one row per movie).

    - id -> int, drop_duplicates("id") (44 rows; the 7 differing copies only
      differ in crew order, so keeping the first is fine)
    - cast: text -> list -> group_cast() (one entry per person)
    - crew: text -> list -> trim_crew()
    Missing or unparseable lists become [].
    """
    df = load_credits(data_dir)
    df["id"] = to_number(df["id"]).astype(int)
    before = len(df)
    df = df.drop_duplicates("id", keep="first").reset_index(drop=True)
    print(f"credits: removed {before - len(df)} duplicate rows, {len(df):,} movies left")
    assert df["id"].is_unique, "duplicate ids remain"

    # .map calls the function once per row: each call gets ONE movie's value
    raw_cast = df["cast"].map(lambda v: parse_literal(v, list) or [])
    df["cast"] = raw_cast.map(group_cast)
    df["crew"] = df["crew"].map(lambda v: parse_literal(v, list) or []).map(trim_crew)

    merged = raw_cast.map(len) - df["cast"].map(len)   # entries removed per movie
    print(f"cast: merged {int(merged.sum()):,} repeated entries in {int((merged > 0).sum()):,} movies")
    return df[["id", "cast", "crew"]]


def group_cast(cast: list[RawCastEntry]) -> list[CastMember]:
    """One entry per person, sorted by billing order.

    Input:  [{"id": 3, "name": .., "gender": 2, "order": 0, "character": "A", ...}, ...]
    Output: [{"id": 3, "name": .., "gender": 2, "order": 0, "characters": ["A", "B"]}, ...]

    - group by person id; order = the person's lowest order; characters in
      the order they appear, without duplicates or empty names
    - sort by (order, position of the person's first entry) - 87 movies have ties
    Expected: 1,207 entries merged in 951 movies.
    """
    people: dict[int, CastMember] = {}   # person id -> that person's grouped entry
    first_seen: dict[int, int] = {}      # person id -> position of their first entry (tie-breaker)

    for position, entry in enumerate(cast):
        person_id = entry["id"]
        if person_id not in people:                 # first time we see this person
            people[person_id] = {
                "id": person_id,
                "name": entry["name"],
                "gender": entry["gender"],
                "order": entry["order"],
                "characters": [],
            }
            first_seen[person_id] = position

        person = people[person_id]
        person["order"] = min(person["order"], entry["order"])   # keep the best billing
        character = entry["character"]
        if character and character not in person["characters"]:
            person["characters"].append(character)

    return sorted(people.values(), key=lambda p: (p["order"], first_seen[p["id"]]))


def trim_crew(crew: list[RawCrewEntry]) -> list[CrewMember]:
    """Keep only CREW_FIELDS (id, name, job, department) for each crew entry."""
    return [{"id": c["id"], "name": c["name"], "job": c["job"], "department": c["department"]}
            for c in crew]


def clean_keywords(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """TODO: keywords.csv -> DataFrame with columns id, keywords.

    - id -> int, drop exact duplicates (987 rows)
    - keywords: parse_literal(v, list) -> [{"id", "name"}]
    """
    raise NotImplementedError


def build_movies(meta: pd.DataFrame, credits: pd.DataFrame,
                 keywords: pd.DataFrame) -> pd.DataFrame:
    """TODO: merge into one row per movie.

    meta.merge(credits, on="id", how="left", validate="one_to_one")
        .merge(keywords, on="id", how="left", validate="one_to_one")
    Missing cast/crew/keywords (e.g. the 1 movie without credits) -> [].
    """
    raise NotImplementedError


def to_movie_documents(movies: pd.DataFrame) -> list[MovieDoc]:
    """One dict per movie, ready for insert_many. _id = TMDB id."""
    docs: list[MovieDoc] = []
    for row in movies.to_dict("records"):
        doc = {k: to_python(v) for k, v in row.items()}
        doc["_id"] = doc.pop("id")
        docs.append(doc)  # type: ignore[arg-type]  # built from the cleaned columns
    assert len(docs) == len({d["_id"] for d in docs}), "duplicate _id in movie documents"
    return docs


def clean_movies(data_dir: Path = DATA_DIR) -> list[MovieDoc]:
    """Run the whole movies pipeline and return the documents."""
    meta = clean_metadata(data_dir)
    movies = build_movies(meta, clean_credits(data_dir), clean_keywords(data_dir))
    docs = to_movie_documents(movies)
    print(f"movies: {len(docs):,} documents (expected {EXPECTED_MOVIES:,})")
    return docs


# =============================================================================
# Users (ratings)
# =============================================================================

def build_link_map(movie_docs: list[MovieDoc], data_dir: Path = DATA_DIR) -> dict[int, int]:
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


def build_genre_lookup(movie_docs: list[MovieDoc]) -> dict[int, list[Genre]]:
    """{tmdbId: [{"id", "name"}, ...]} - genres copied into every rating."""
    return {d["_id"]: d["genres"] for d in movie_docs}


def iter_user_documents(link_map: dict[int, int], genre_lookup: dict[int, list[Genre]],
                        data_dir: Path = DATA_DIR, small: bool = False,
                        stats: Counter[str] | None = None) -> Iterator[UserDoc]:
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

def to_python(value: Any) -> Any:
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