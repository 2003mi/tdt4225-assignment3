"""Extra step 2: where the duplicate ids come from, and cleaning links.csv.

The main EDA found 29 duplicated ids in movies_metadata.csv, 30 duplicate
tmdbIds in links.csv and 95 links whose imdb id disagrees with the metadata.
This step checks two things:
  1. Are the duplicated metadata ids the same movies as the shared tmdbIds?
     The dataset was built by fetching TMDB data per link row, so two links
     to one tmdbId would produce two metadata rows.
  2. After dropping links without a tmdbId and links whose imdb id disagrees
     with the metadata, are any tmdbIds still shared? If not, links.csv is a
     strict 1:1 mapping and ratings can be translated movieId -> tmdbId.

Finding: 29 / 29 / 29 - the metadata duplicates are exactly the shared
tmdbIds. After dropping 219 links without tmdbId and 95 imdb mismatches,
45,529 links remain with 0 shared tmdbIds.
"""
import pandas as pd

from ..load import to_number
from .common import section


def run(raw_links: pd.DataFrame, meta: pd.DataFrame) -> None:
    """raw_links: links.csv as text. meta: metadata with integer ids (duplicates kept)."""
    section("Extra: duplicate ids vs. links, and cleaned links")
    links = raw_links.copy()
    links["tmdbId"] = to_number(links["tmdbId"]).astype("Int64")
    links["imdb_num"] = to_number(links["imdbId"])

    dup_meta = set(meta.loc[meta["id"].duplicated(), "id"])
    dup_links = set(links.loc[links["tmdbId"].notna() & links["tmdbId"].duplicated(), "tmdbId"])
    print(f"Duplicated ids in metadata: {len(dup_meta)}   Shared tmdbIds in links: {len(dup_links)}   "
          f"In both: {len(dup_meta & dup_links)}")

    imdb_num = to_number(meta["imdb_id"].str.removeprefix("tt"))
    imdb_of = meta.assign(imdb_num=imdb_num).drop_duplicates("id").set_index("id")["imdb_num"]
    expected = links["tmdbId"].map(imdb_of)
    missing_tmdb = links["tmdbId"].isna()
    mismatch = ~missing_tmdb & expected.notna() & (expected != links["imdb_num"])
    good = links[~missing_tmdb & ~mismatch]
    print(f"Links: {len(links):,}   without tmdbId: {int(missing_tmdb.sum()):,}   "
          f"imdb mismatch: {int(mismatch.sum()):,}   kept: {len(good):,}")
    print(f"Shared tmdbIds left after dropping those: {int(good['tmdbId'].duplicated().sum())}")
    linked = set(good["tmdbId"].dropna().astype(int))
    print(f"Metadata movies left without a link: {len(set(meta['id']) - linked):,}")
