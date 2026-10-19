"""Rules for links.csv, shared by the EDA (src/eda/lost_ratings.py) and the
cleaning step (src/clean.py: build_link_map).

Ratings use MovieLens movieIds; links.csv translates them to TMDB ids. Every
link gets one reason:
  - ok                     - tmdbId exists in the metadata and the imdb ids agree
  - no tmdbId              - the MovieLens movie had no TMDB match
  - tmdbId not in metadata - points at a TMDB movie that was never fetched
  - imdb mismatch          - points at a different film (usually same title)
Only "ok" links are used.
"""
import pandas as pd

from .load import to_number

REASONS = ["ok", "no tmdbId", "tmdbId not in metadata", "imdb mismatch"]


def classify_links(raw_links: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Return links.csv with integer ids, imdb_num and a 'reason' column.

    raw_links: links.csv as text. meta: any table with columns "id" (TMDB id)
    and "imdb_id" ("tt0114709"); duplicate ids are ignored.
    """
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
