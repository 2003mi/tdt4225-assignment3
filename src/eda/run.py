"""Runs the whole EDA in order:

  0. broken_rows    - repair the 3 records split by a line break (first, so no
                      movie is missing from the rest of the EDA)
  core              - one section per CSV: metadata, credits, keywords, links,
                      ratings
  1. small_values   - budgets/revenues under $1,000
  2. links_check    - where the duplicate ids come from, cleaned links
  3. lost_ratings   - ratings lost when linking to movies, by reason
  4. credits_dups   - credits rows that share an id but differ
  5. quality        - placeholders, languages, same film twice, people
                      listed twice in one movie
  6. ratings_sorted - is ratings.csv sorted and grouped by userId?

Each CSV is loaded once here and passed to the steps that need it.
"""
from pathlib import Path

from ..load import load_credits, load_keywords, load_links, load_metadata
from ..paths import DATA_DIR
from . import (broken_rows, core, credits_dups, links_check, lost_ratings, quality,
               ratings_sorted, small_values)
from .common import PLOT_DIR, banner, style


def run_eda(data_dir: Path = DATA_DIR, small: bool = False, include_ratings: bool = True) -> None:
    style()

    banner("STEP 0: REPAIR BROKEN ROWS")
    meta_raw = broken_rows.run(load_metadata(data_dir), data_dir)

    banner("MAIN EDA")
    meta = core.eda_metadata(meta_raw)
    movie_ids = set(meta["id"])
    titles = meta.drop_duplicates("id").set_index("id")["title"]

    raw_credits = load_credits(data_dir)
    credits = core.eda_credits(raw_credits, movie_ids)
    core.eda_keywords(load_keywords(data_dir), movie_ids)
    raw_links = load_links(data_dir)
    link_map = core.eda_links(raw_links, meta)
    movie_counts = core.eda_ratings(data_dir, small, link_map) if include_ratings else None

    banner("EXTRA CHECKS")
    small_values.run(meta)
    links_check.run(raw_links, meta)
    if movie_counts is not None:
        lost_ratings.run(raw_links, meta, movie_counts)
    credits_dups.run(raw_credits, titles)
    quality.run(meta, credits, titles)
    if include_ratings:
        ratings_sorted.run(data_dir, small)

    print(f"\nPlots saved to {PLOT_DIR}")
