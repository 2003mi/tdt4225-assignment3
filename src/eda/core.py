"""Main EDA: one function per CSV file.

Every CSV is read with all columns as text (see src/load.py), so nothing is
silently converted and we can count exactly which values fail to become
numbers, dates or lists. For each file we look at four things:

1. Structure   - row/column counts, whether the id column is valid, duplicates.
2. Missing     - empty fields (NaN), but also values that *look* filled in and
                 really mean "unknown": 0 for budget/revenue/runtime/votes,
                 [] for list columns, gender 0 for cast.
3. Content     - distributions of the important columns, and the nested
                 objects inside the Python-literal columns (what keys they
                 have, how many items per movie).
4. Joins       - whether ids in one file actually exist in the others, since
                 the Part 2 queries combine metadata, credits and ratings.

Numbers that directly matter for a Part 2 query (e.g. how many directors have
>= 5 movies for Q1) are printed so the query results can be sanity-checked.

The functions take already-loaded DataFrames (see run.py); movies_metadata.csv
is passed in after the broken rows have been repaired (broken_rows.py).
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..load import METADATA_NESTED, iter_ratings, parse_literal, to_number
from .common import (BAR, GENDER, INK_2, hbar, hist, log_bins, pct, save_plot,
                     section, show, subsection)


def _check_ids(raw: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Check the id column, which every join between the files depends on.

    - Non-numeric ids: in movies_metadata.csv these are the rows where a
      broken quote in the CSV shifted all columns one step, so a date ends
      up in the id column.
    - Duplicate ids: counted both as "same id" and "entire row identical".
      Exact copies can simply be dropped; rows that share an id but differ
      need a rule for which one to keep.

    Returns (rows with a valid id, with id converted to int; mask of bad rows).
    """
    ids = to_number(raw["id"])
    bad = ids.isna()
    print(f"Rows: {len(raw):,}   Columns: {raw.shape[1]}")
    print(f"Rows with non-numeric id: {bad.sum():,}")
    df = raw.loc[~bad].copy()
    df["id"] = ids[~bad].astype("int64")
    n_dup_ids = df.loc[df["id"].duplicated(keep=False), "id"].nunique()
    n_extra = int(df["id"].duplicated().sum())
    n_exact = int(df.duplicated().sum())
    print(f"Duplicate ids in {name}: {n_dup_ids:,} ids occur more than once "
          f"({n_extra:,} extra rows, {n_exact:,} of them exact copies of another row)")
    return df, bad


# ----------------------------------------------------------------------------
# movies_metadata.csv
# ----------------------------------------------------------------------------

NUMERIC = ["budget", "revenue", "runtime", "vote_average", "vote_count", "popularity"]


def eda_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    """movies_metadata.csv - one row per movie, the main file.

    Structure
      - The rows split by a line break have already been repaired
        (broken_rows.py), so "non-numeric id" should be 0 here; any that
        remain are printed. Duplicate ids are counted. For ids that occur more than once but
        are not exact copies, we list which columns differ between the copies.

    Missing values
      - "missing" = empty field in the CSV.
      - "0 or empty []" = a value is present but means unknown. TMDB stores an
        unknown budget/revenue/runtime as 0, and no genres as "[]". These
        have to become null before insert, otherwise medians/totals are wrong.

    Numeric columns (budget, revenue, runtime, vote_average, vote_count, popularity)
      - Stored as text in the CSV. We count values that fail to parse, zeros,
        and min/median/mean/max of the values > 0.
      - Extra checks: budget/revenue under 1000 USD (probably entered in
        millions or placeholders), runtime over 300 min, how many movies
        have both budget and revenue, and how many pass the vote_count
        thresholds used in Q7 (>= 50) and Q8 (>= 100).

    Release dates
      - Missing/unparseable dates, the date range, and movies per decade.
        Q5 and Q6 group by decade, so movies without a date drop out there.

    Categorical columns
      - status, adult, video and original_language value counts. Q9 filters
        on original_language != "en".

    Nested columns (genres, production_companies, production_countries,
    spoken_languages, belongs_to_collection)
      - Parsed with ast.literal_eval (single quotes, so not JSON). We count
        parse failures and empty lists, and list which keys the objects
        contain - e.g. production_companies has no country, which matters
        for how Q9 is interpreted.
      - Genres: count per genre, both in any position and as the primary
        (first) genre used in Q5, plus genres per movie (Q3, Q10).
      - Collections: movies per collection and how many have >= 3 (Q4).
      - Movies with a US production country, and the non-English + US
        subset that Q9 works on.

    Text columns
      - Missing overview/tagline, and how many mention "noir" (Q7).

    Returns the metadata with numeric columns converted, dates parsed and
    nested columns as Python lists/dicts.
    """
    section("movies_metadata.csv (after repairing the broken rows)")
    df, bad = _check_ids(raw, "metadata")

    if bad.any():
        subsection("Malformed rows (id is not a number — the columns are shifted)")
        show(raw.loc[bad, ["id", "title", "budget", "release_date"]].fillna("NaN"))

    # Rows that share an id but are not identical: which columns differ?
    dup = df[df["id"].duplicated(keep=False)]
    differing: Counter[str] = Counter()
    for _, group in dup.groupby("id"):
        if len(group.drop_duplicates()) > 1:
            for col in group.columns:
                if group[col].nunique(dropna=False) > 1:
                    differing[col] += 1
    if differing:
        print("Columns that differ between rows sharing an id:",
              ", ".join(f"{c} ({n})" for c, n in differing.most_common()))

    n = len(df)

    # --- Missing values (on the raw strings), plus "unknown" encoded as 0 or []
    subsection("Missing values (rows with a valid id)")
    rows = []
    for col in raw.columns:
        missing = int(df[col].isna().sum())
        if col in NUMERIC:
            zero_or_empty = int((to_number(df[col]) == 0).sum())
        elif col in METADATA_NESTED:
            zero_or_empty = int(df[col].isin(["[]", "{}"]).sum())
        else:
            zero_or_empty = 0
        rows.append([col, missing, pct(missing, n), zero_or_empty,
                     pct(missing + zero_or_empty, n)])
    missing_tbl = pd.DataFrame(rows, columns=["column", "missing", "missing %",
                                              "0 or empty []", "total unknown %"])
    show(missing_tbl)
    unknown = (missing_tbl.set_index("column")["missing"]
               + missing_tbl.set_index("column")["0 or empty []"]) / n * 100
    unknown = unknown[unknown > 0].sort_values(ascending=False)
    hbar(list(unknown.index), list(unknown.values),
          "Share of movies with a missing or unknown value",
          "% of movies (missing, 0, or empty list)", "metadata_missing.png",
          fmt="{:.1f}%")

    # --- Types: convert and count values that do not parse
    subsection("Numeric columns (stored as text in the CSV)")
    rows = []
    for col in NUMERIC:
        values = to_number(df[col])
        unparseable = int((df[col].notna() & values.isna()).sum())
        pos = values[values > 0]
        rows.append([col, int(values.notna().sum()), unparseable, int((values == 0).sum()),
                     pos.min() if len(pos) else None, pos.median() if len(pos) else None,
                     pos.mean() if len(pos) else None, pos.max() if len(pos) else None])
        df[col] = values
    show(rows, headers=["column", "valid", "unparseable", "zeros",
                        "min (>0)", "median (>0)", "mean (>0)", "max"])

    small_budget = int(((df["budget"] > 0) & (df["budget"] < 1000)).sum())
    small_revenue = int(((df["revenue"] > 0) & (df["revenue"] < 1000)).sum())
    print(f"Suspiciously small values (0 < x < 1000 USD): budget {small_budget}, revenue {small_revenue}")
    print(f"Runtime over 300 min: {int((df['runtime'] > 300).sum())}")
    both = (df["budget"] > 0) & (df["revenue"] > 0)
    print(f"Movies with both budget and revenue known: {int(both.sum()):,} ({pct(both.sum(), n)})")
    for t in (50, 100):
        k = int((df["vote_count"] >= t).sum())
        print(f"Movies with vote_count >= {t}: {k:,} ({pct(k, n)})")

    # --- Release dates
    subsection("Release dates")
    dates = pd.to_datetime(df["release_date"], errors="coerce", format="%Y-%m-%d")
    unparseable = int((df["release_date"].notna() & dates.isna()).sum())
    print(f"Missing: {int(df['release_date'].isna().sum())}   Unparseable: {unparseable}   "
          f"Range: {dates.min():%Y-%m-%d} -> {dates.max():%Y-%m-%d}")
    df["release_date"] = dates
    decade = (dates.dt.year // 10 * 10).dropna().astype(int)
    per_decade = decade.value_counts().sort_index()
    show([[f"{d}s", c, pct(c, n)] for d, c in per_decade.items()],
         headers=["decade", "movies", "share"])

    per_year = dates.dt.year.dropna().astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(per_year.index, per_year.values, color=BAR, linewidth=2)
    ax.set_title("Movies released per year")
    ax.set_ylabel("Movies")
    ax.set_ylim(bottom=0)
    save_plot(fig, "metadata_releases_per_year.png")

    # --- Distributions
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    rt = df["runtime"]
    hist(axes[0, 0], rt[(rt > 0) & (rt <= 300)], 60, "Runtime", "Minutes (0 < runtime <= 300)")
    va = df.loc[df["vote_count"] > 0, "vote_average"]
    hist(axes[0, 1], va, 20, "vote_average", "Rating 0–10 (movies with >= 1 vote)")
    vc = df.loc[df["vote_count"] > 0, "vote_count"]
    hist(axes[1, 0], vc, log_bins(vc), "vote_count", "Votes (log scale)", log_x=True)
    rev = df.loc[df["revenue"] > 0, "revenue"]
    if len(rev):
        hist(axes[1, 1], rev, log_bins(rev), "revenue", "USD (log scale, revenue > 0)",
              log_x=True)
    fig.tight_layout()
    save_plot(fig, "metadata_distributions.png")

    # --- Categorical columns
    subsection("Categorical columns")
    for col in ("status", "adult", "video"):
        vc_ = df[col].fillna("NaN").value_counts()
        print(f"{col}: " + ", ".join(f"{k}={v:,}" for k, v in vc_.items()))
    langs = df["original_language"].value_counts()
    print(f"original_language: {langs.size} distinct, top 10:")
    show([[k, v, pct(v, n)] for k, v in langs.head(10).items()],
         headers=["language", "movies", "share"])

    # --- Nested columns
    subsection("Nested columns (Python-literal strings)")
    rows = []
    for col, typ in METADATA_NESTED.items():
        parsed = df[col].map(lambda v, typ=typ: parse_literal(v, typ))
        failed = int((df[col].notna() & parsed.isna()).sum())
        keys = set()
        for obj in parsed.dropna():
            for item in (obj if isinstance(obj, list) else [obj]):
                keys.update(item.keys())
        empty = int(parsed.map(lambda o: o is not None and len(o) == 0).sum())
        rows.append([col, int(parsed.notna().sum()), empty, failed, ", ".join(sorted(keys))])
        df[col] = parsed
    show(rows, headers=["column", "parsed", "empty", "failed to parse", "keys in objects"])

    # Genres
    names = df["genres"].map(lambda g: [x.get("name") for x in g] if g else [])
    genre_counts = Counter(name for lst in names for name in lst)
    primary = names.map(lambda lst: lst[0] if lst else None).value_counts()
    print(f"\nGenres: {len(genre_counts)} distinct. Genres per movie: "
          + ", ".join(f"{k}={v:,}" for k, v in names.map(len).value_counts().sort_index().items()))
    show([[g, c, int(primary.get(g, 0))] for g, c in genre_counts.most_common()],
         headers=["genre", "movies (any position)", "movies (primary = first)"])
    top = genre_counts.most_common()
    hbar([g for g, _ in top], [c for _, c in top], "Movies per genre",
          "Movies (a movie counts once for each of its genres)", "metadata_genres.png")

    # Collections
    coll = df["belongs_to_collection"].map(lambda c: c.get("name") if c else None).dropna()
    sizes = coll.value_counts()
    print(f"\nCollections: {len(coll):,} movies belong to {sizes.size:,} collections; "
          f"{int((sizes >= 3).sum()):,} collections have >= 3 movies")
    show([[k, v] for k, v in sizes.head(5).items()], headers=["largest collections", "movies"])

    # Countries / companies / languages
    countries = df["production_countries"].map(
        lambda lst: {c.get("iso_3166_1") for c in lst} if lst else set())
    has_us = countries.map(lambda s: "US" in s)
    non_en = df["original_language"].notna() & (df["original_language"] != "en")
    companies = {c.get("name") for lst in df["production_companies"].dropna() for c in lst}
    print(f"\nProduction countries include US: {int(has_us.sum()):,} ({pct(has_us.sum(), n)})")
    print(f"Distinct production companies: {len(companies):,}")
    print(f"original_language != 'en' AND a US production country: {int((non_en & has_us).sum()):,}")

    # Text columns
    subsection("Text columns")
    for col in ("overview", "tagline"):
        empty = df[col].isna() | (df[col].str.strip() == "")
        print(f"{col}: {int(empty.sum()):,} missing/empty ({pct(empty.sum(), n)})")
    text = (df["overview"].fillna("") + " " + df["tagline"].fillna("")).str.lower()
    print(f"overview/tagline containing 'noir': {int(text.str.contains('noir', regex=False).sum()):,}")

    return df


# ----------------------------------------------------------------------------
# credits.csv
# ----------------------------------------------------------------------------

def eda_credits(raw: pd.DataFrame, movie_ids: set[int]) -> pd.DataFrame:
    """credits.csv - one row per movie with its full cast and crew lists.

    Structure
      - Valid and duplicate ids (same checks as metadata).
      - cast and crew are Python-literal lists; we count parse failures,
        empty lists, and the median/max list length.
      - Join check: credit ids missing from metadata, and metadata movies
        that have no credits row.

    Returns the credits with cast/crew parsed into lists, one row per movie.

    People (counted once per movie, after dropping duplicate movie ids)
      - Distinct actors, and how many have >= 10 movies (Q3).
      - Distinct directors (crew with job == "Director"), and how many have
        >= 5 movies (Q1). People are counted by their TMDB person id, not by
        name, since different people can share a name.
      - Directors per movie (0, 1, 2+). Movies with several directors count
        towards each of them in Q1/Q8; movies with none drop out.
      - Cast entries without an "order" field, since Q6 uses order to pick
        the top-5 billed actors.
      - The 10 most common crew jobs.

    Cast gender
      - TMDB uses 0 = unknown, 1 = female, 2 = male. Shares are shown for
        the full cast and for the top-5 billed per movie (the group Q6 is
        about). A high unknown share means Q6 rests on fewer actors than
        the cast size suggests.
    """
    section("credits.csv")
    df, _ = _check_ids(raw, "credits")

    df["cast"] = df["cast"].map(lambda v: parse_literal(v, list))
    df["crew"] = df["crew"].map(lambda v: parse_literal(v, list))
    for col in ("cast", "crew"):
        failed = int((raw.loc[df.index, col].notna() & df[col].isna()).sum())
        sizes = df[col].map(lambda lst: len(lst) if lst else 0)
        print(f"{col}: failed to parse {failed}, empty {int((sizes == 0).sum()):,}, "
              f"median size {sizes.median():.0f}, max {sizes.max():,}")

    ids = set(df["id"])
    print(f"Credit ids not in metadata: {len(ids - movie_ids):,}   "
          f"Metadata movies without credits: {len(movie_ids - ids):,}")

    df = df.drop_duplicates("id")
    actor_movies: Counter[int] = Counter()       # person id -> movies
    director_movies: Counter[int] = Counter()    # person id -> movies
    gender_all: Counter[int] = Counter()         # gender code -> cast entries
    gender_top5: Counter[int] = Counter()
    jobs: Counter[str] = Counter()               # crew job -> entries
    directors_per_movie: Counter[int] = Counter()  # 0, 1, 2 (= 2 or more) -> movies
    missing_order = 0
    for cast, crew in zip(df["cast"], df["crew"]):
        cast, crew = cast or [], crew or []
        actor_movies.update({p.get("id") for p in cast})
        gender_all.update(p.get("gender") for p in cast)
        missing_order += sum("order" not in p for p in cast)
        top5 = sorted(cast, key=lambda p: p.get("order", float("inf")))[:5]
        gender_top5.update(p.get("gender") for p in top5)
        directors = {p.get("id") for p in crew if p.get("job") == "Director"}
        directors_per_movie[min(len(directors), 2)] += 1
        director_movies.update(directors)
        jobs.update(p.get("job") for p in crew)

    subsection("People")
    print(f"Distinct actors: {len(actor_movies):,} "
          f"(with >= 10 movies: {sum(c >= 10 for c in actor_movies.values()):,})")
    print(f"Distinct directors: {len(director_movies):,} "
          f"(with >= 5 movies: {sum(c >= 5 for c in director_movies.values()):,})")
    print("Directors per movie: " + ", ".join(
        f"{'2+' if k == 2 else k}={v:,}" for k, v in sorted(directors_per_movie.items())))
    print(f"Cast entries without 'order': {missing_order:,}")
    show(jobs.most_common(10), headers=["crew job", "entries"])

    subsection("Cast gender (0 = unknown, 1 = female, 2 = male)")
    rows = []
    for label, counts in (("All cast", gender_all), ("Top-5 billed", gender_top5)):
        total = sum(counts.values())
        rows.append([label, total] + [pct(counts.get(g, 0), total) for g in (1, 2, 0)])
    show(rows, headers=["", "entries", "female", "male", "unknown"])

    fig, ax = plt.subplots(figsize=(7, 1.9))
    for y, (label, counts) in enumerate((("All cast", gender_all), ("Top-5 billed", gender_top5))):
        total = sum(counts.values()) or 1
        left = 0.0
        for g in (1, 2, 0):
            name, color = GENDER[g]
            share = 100 * counts.get(g, 0) / total
            ax.barh(y, share, left=left, color=color, edgecolor="white", linewidth=1.5,
                    height=0.6, label=name if y == 0 else None)
            if share >= 6:
                ax.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                        fontsize=8, color="white")
            left += share
    ax.set_yticks([0, 1], ["All cast", "Top-5 billed"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of cast entries")
    ax.grid(axis="y", visible=False)
    ax.set_title("Cast gender", pad=22)
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              borderaxespad=0.2, fontsize=8)
    save_plot(fig, "credits_gender.png")
    return df[["id", "cast", "crew"]]


# ----------------------------------------------------------------------------
# keywords.csv
# ----------------------------------------------------------------------------

def eda_keywords(raw: pd.DataFrame, movie_ids: set[int]) -> None:
    """keywords.csv - one row per movie with a list of plot keywords.

    - Valid and duplicate ids.
    - Parse failures and movies with an empty keyword list (no keywords is
      stored as "[]", not as an empty field, so it never shows as missing).
    - Number of distinct keywords and the 10 most common ones.
    - Join check: keyword ids that are not in metadata.
    No Part 2 query uses keywords directly, but they are part of the dataset
    and decide whether keywords are worth embedding in the movie documents.
    """
    section("keywords.csv")
    df, _ = _check_ids(raw, "keywords")
    parsed = df["keywords"].map(lambda v: parse_literal(v, list))
    failed = int((df["keywords"].notna() & parsed.isna()).sum())
    sizes = parsed.map(lambda lst: len(lst) if lst else 0)
    counts = Counter(k.get("name") for lst in parsed.dropna() for k in lst)
    print(f"Failed to parse: {failed}   Empty keyword list: {int((sizes == 0).sum()):,} "
          f"({pct((sizes == 0).sum(), len(df))})   Distinct keywords: {len(counts):,}")
    print(f"Keyword ids not in metadata: {len(set(df['id']) - movie_ids):,}")
    show(counts.most_common(10), headers=["keyword", "movies"])


# ----------------------------------------------------------------------------
# links.csv
# ----------------------------------------------------------------------------

def eda_links(raw: pd.DataFrame, meta: pd.DataFrame) -> dict[int, int]:
    """links.csv - maps MovieLens movieId (used in ratings) to tmdbId (the
    id in metadata) and imdbId. It is the only bridge between the ratings
    and the movies, so its quality decides how much of Q10 is usable.

    The full links.csv is always used, also with --small: it covers every
    MovieLens movie, while links_small.csv only covers the ~9k movies in
    ratings_small.csv.

    - Missing tmdbId / imdbId: those MovieLens movies cannot be joined.
    - Duplicate movieId or tmdbId: a duplicate would make one rating count
      for two movies, or two MovieLens movies share one TMDB movie. This
      decides whether movieId can be stored as a single field on a movie.
    - How many tmdbIds exist in the metadata, and how many metadata movies
      have no link at all.
    - imdb cross-check: links.csv has its own imdbId ("0114709") and the
      metadata has imdb_id ("tt0114709"). If a link's tmdbId points to a
      movie with a different imdb id, that link most likely points to the
      wrong movie, and its ratings would be counted for the wrong film.

    Returns a dict {movieId: tmdbId} for links that point at a known movie,
    used to measure join coverage in eda_ratings.
    """
    section("links.csv")
    movie_id = to_number(raw["movieId"])
    tmdb = to_number(raw["tmdbId"])
    n = len(raw)
    movie_ids = set(meta["id"])
    print(f"Rows: {n:,}")
    print(f"Missing tmdbId: {int(tmdb.isna().sum()):,}   Missing imdbId: {int(raw['imdbId'].isna().sum()):,}")
    print(f"Duplicate movieId: {int(movie_id.duplicated().sum()):,}   "
          f"Duplicate tmdbId: {int(tmdb.dropna().duplicated().sum()):,}")

    in_meta = tmdb.isin(movie_ids)
    linked = set(tmdb[in_meta].astype(int))
    print(f"tmdbId found in metadata: {int(in_meta.sum()):,} ({pct(in_meta.sum(), n)})")
    print(f"Metadata movies with no link (cannot get any ratings): {len(movie_ids - linked):,}")

    # tmdbIds shared by several MovieLens movies
    shared = raw.assign(tmdb=tmdb).dropna(subset=["tmdb"])
    shared = shared[shared["tmdb"].duplicated(keep=False)].sort_values("tmdb")
    if len(shared):
        subsection("tmdbIds used by more than one movieId (first 10)")
        show(shared[["movieId", "imdbId", "tmdbId"]].head(10))

    # imdb cross-check between links.csv and movies_metadata.csv
    meta_imdb = (meta.drop_duplicates("id").set_index("id")
                 .assign(imdb_num=lambda m: to_number(m["imdb_id"].str.removeprefix("tt"))))
    ok = movie_id.notna() & in_meta
    check = pd.DataFrame({
        "movieId": movie_id[ok].astype(int),
        "tmdbId": tmdb[ok].astype(int),
        "imdbId (links)": to_number(raw.loc[ok, "imdbId"]),
    })
    check["imdb_id (metadata)"] = check["tmdbId"].map(meta_imdb["imdb_id"])
    check["title"] = check["tmdbId"].map(meta_imdb["title"])
    known = check["imdbId (links)"].notna() & check["tmdbId"].map(meta_imdb["imdb_num"]).notna()
    mismatch = known & (check["imdbId (links)"] != check["tmdbId"].map(meta_imdb["imdb_num"]))
    print(f"Linked movies where the imdb id in links.csv and metadata disagree: "
          f"{int(mismatch.sum()):,} of {int(known.sum()):,} comparable")
    if mismatch.any():
        bad = check[mismatch].copy()
        bad["imdbId (links)"] = "tt" + bad["imdbId (links)"].astype(int).astype(str).str.zfill(7)
        bad["title"] = bad["title"].str.slice(0, 40)
        show(bad.head(10))

    return dict(zip(check["movieId"], check["tmdbId"]))


# ----------------------------------------------------------------------------
# ratings.csv (read in chunks)
# ----------------------------------------------------------------------------

def eda_ratings(data_dir: Path, small: bool, link_map: dict[int, int]) -> pd.Series:
    """ratings.csv - one row per (user, movie) rating from MovieLens.

    The full file has ~26M rows, so it is read in chunks of 1M rows and only
    running totals are kept (value counts per rating, per user and per movie).

    - Totals: number of ratings, users and rated movies, and the timestamp
      range (timestamps are Unix seconds).
    - Duplicate (userId, movieId) pairs - a user should rate a movie once.
      The file is sorted by userId, so a user's rows can only be split across
      a chunk boundary; those rows are carried into the next chunk so the
      check stays exact. If the file turns out not to be sorted, this is
      reported and the duplicate count is only a lower bound.
    - Rating values: MovieLens uses 0.5 to 5.0 in half-star steps (unlike
      TMDB's 0-10 vote_average), so any other value would be an error.
    - Activity: ratings per user and per movie (min/median/mean/max), and
      how many users have >= 20 ratings (the Q10 threshold).
    - Join coverage: how many rated movies, and how many ratings, can be
      followed through links to a movie in metadata. Ratings that cannot
      be joined have no genres and are lost for Q10.

    Returns the number of ratings per movieId (used by lost_ratings.py).
    """
    name = "ratings_small.csv" if small else "ratings.csv"
    section(name)
    total = 0
    rating_counts = pd.Series(dtype="float64")
    user_counts = pd.Series(dtype="float64")
    movie_counts = pd.Series(dtype="float64")
    ts_min, ts_max = None, None
    duplicates, sorted_by_user, carry, last_user = 0, True, None, None

    for i, chunk in enumerate(iter_ratings(data_dir, small)):
        total += len(chunk)
        print(f"  read {total:,} rows...", end="\r")
        rating_counts = rating_counts.add(chunk["rating"].value_counts(), fill_value=0)
        user_counts = user_counts.add(chunk["userId"].value_counts(), fill_value=0)
        movie_counts = movie_counts.add(chunk["movieId"].value_counts(), fill_value=0)
        lo, hi = chunk["timestamp"].min(), chunk["timestamp"].max()
        ts_min = lo if ts_min is None else min(ts_min, lo)
        ts_max = hi if ts_max is None else max(ts_max, hi)

        # Duplicate (userId, movieId) check. The file is sorted by userId, so a
        # user's rows can only span a chunk boundary; carry them over.
        if not chunk["userId"].is_monotonic_increasing or (
                last_user is not None and chunk["userId"].iloc[0] < last_user):
            sorted_by_user = False
        pairs = chunk[["userId", "movieId"]]
        if carry is not None:
            pairs = pd.concat([carry, pairs])
            duplicates -= int(carry.duplicated().sum())  # already counted
        duplicates += int(pairs.duplicated().sum())
        last_user = chunk["userId"].iloc[-1]
        carry = chunk.loc[chunk["userId"] == last_user, ["userId", "movieId"]]
    print(" " * 40, end="\r")

    user_counts = user_counts.astype(int)
    movie_counts = movie_counts.astype(int)
    print(f"Ratings: {total:,}   Users: {user_counts.size:,}   Movies rated: {movie_counts.size:,}")
    print(f"Time range: {pd.to_datetime(ts_min, unit='s'):%Y-%m-%d} -> "
          f"{pd.to_datetime(ts_max, unit='s'):%Y-%m-%d}")
    note = "" if sorted_by_user else " (file not sorted by userId — check is incomplete)"
    print(f"Duplicate (userId, movieId) pairs: {duplicates:,}{note}")

    subsection("Rating values")
    show([[f"{r:.1f}", int(c), pct(c, total)] for r, c in rating_counts.sort_index().items()],
         headers=["rating", "count", "share"])

    subsection("Activity")
    rows = []
    for label, s in (("ratings per user", user_counts), ("ratings per movie", movie_counts)):
        rows.append([label, int(s.min()), float(s.median()), float(s.mean()), int(s.max())])
    show(rows, headers=["", "min", "median", "mean", "max"])
    print(f"Users with >= 20 ratings: {int((user_counts >= 20).sum()):,} "
          f"({pct((user_counts >= 20).sum(), user_counts.size)})")

    subsection("Joining ratings -> links -> metadata")
    linked = movie_counts.index.isin(list(link_map))
    print(f"Rated movies that map to a metadata movie: {int(linked.sum()):,} of {movie_counts.size:,}")
    print(f"Ratings that can be joined to metadata: {int(movie_counts[linked].sum()):,} "
          f"({pct(movie_counts[linked].sum(), total)})")

    fig, ax = plt.subplots(figsize=(7, 3))
    labels = [f"{r:g}" for r in rating_counts.index]
    ax.bar(labels, rating_counts.values / 1e6 if total > 1e6 else rating_counts.values,
           color=BAR, width=0.72)
    ax.set_title("Rating values")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Ratings (millions)" if total > 1e6 else "Ratings")
    ax.grid(axis="x", visible=False)
    save_plot(fig, "ratings_values.png")

    fig, ax = plt.subplots(figsize=(7, 3))
    hist(ax, user_counts, log_bins(user_counts), "Ratings per user", "Ratings (log scale)",
          log_x=True)
    ax.set_ylabel("Users")
    ax.axvline(20, color=INK_2, linewidth=1, linestyle="--")
    ax.text(20, ax.get_ylim()[1] * 0.97, " >= 20 ratings (Q10) ", color=INK_2, fontsize=8,
            va="top", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    save_plot(fig, "ratings_per_user.png")
    return movie_counts

