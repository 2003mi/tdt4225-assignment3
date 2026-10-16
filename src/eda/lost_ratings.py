"""Extra step 3: how many ratings are lost when linking them to movies, and why.

Ratings use MovieLens movieIds; links.csv translates them to TMDB ids. Every
link gets one reason:
  - ok                     - tmdbId exists in the metadata and the imdb ids agree
  - no tmdbId              - the MovieLens movie had no TMDB match
  - tmdbId not in metadata - points at a TMDB movie that was never fetched
  - imdb mismatch          - points at a different film (usually same title)
and the ratings behind each reason are added up. Links without a tmdbId are
also checked against the metadata's imdb_id, the only honest way to recover
them.

Finding (full ratings.csv): 0.209% of the ratings are lost - 13,503 with no
tmdbId, 29,204 pointing outside the metadata, 11,729 behind wrong links.
None of the 219 links without a tmdbId can be recovered via imdb. 45,368
links and 25,969,853 ratings remain. The imdb rule also drops a few links
that are a second MovieLens entry for the same film (e.g. Pokemon 3, tmdb
10991); without MovieLens titles these cannot be told apart from wrong links.
"""
import pandas as pd

from ..load import to_number
from .common import section, show

REASONS = ["ok", "no tmdbId", "tmdbId not in metadata", "imdb mismatch"]


def classify_links(raw_links: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Return links.csv with integer ids, imdb_num and a 'reason' column."""
    links = raw_links.copy()
    links["movieId"] = to_number(links["movieId"]).astype(int)
    links["tmdbId"] = to_number(links["tmdbId"]).astype("Int64")
    links["imdb_num"] = to_number(links["imdbId"])

    m = meta.drop_duplicates("id")
    imdb_of = pd.Series(to_number(m["imdb_id"].str.removeprefix("tt")).values, index=m["id"])
    expected = links["tmdbId"].map(imdb_of)

    links["reason"] = "ok"
    links.loc[links["tmdbId"].isna(), "reason"] = "no tmdbId"
    links.loc[links["tmdbId"].notna() & ~links["tmdbId"].isin(m["id"]), "reason"] = "tmdbId not in metadata"
    links.loc[expected.notna() & (expected != links["imdb_num"]), "reason"] = "imdb mismatch"
    return links


def run(raw_links: pd.DataFrame, meta: pd.DataFrame, movie_counts: pd.Series) -> None:
    """movie_counts: ratings per movieId, from core.eda_ratings."""
    section("Extra: ratings lost when linking to movies")
    links = classify_links(raw_links, meta)
    links["ratings"] = links["movieId"].map(movie_counts).fillna(0).astype(int)
    total = int(movie_counts.sum())

    rows = []
    for reason in REASONS:
        part = links[links["reason"] == reason]
        rows.append([reason, len(part), int(part["ratings"].sum()),
                     f"{100 * part['ratings'].sum() / total:.3f}%" if total else "-"])
    show(rows, headers=["reason", "links", "ratings", "share of ratings"])

    lost = links[links["reason"] != "ok"]
    print(f"Lost in total: {len(lost):,} links, {int(lost['ratings'].sum()):,} ratings")
    print(f"Rated movieIds missing from links.csv: "
          f"{int((~movie_counts.index.isin(links['movieId'])).sum())}")

    no_tmdb = links[links["reason"] == "no tmdbId"]
    known_imdb = to_number(meta["imdb_id"].str.removeprefix("tt"))
    print(f"Links without tmdbId whose imdb id exists in metadata (recoverable): "
          f"{int(no_tmdb['imdb_num'].isin(known_imdb).sum())} of {len(no_tmdb)}")
