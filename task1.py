"""Part 1: EDA, cleaning and inserting the Movies dataset.

    python task1.py               # EDA, then clean + insert (full dataset)
    python task1.py --skip-eda    # only clean + insert
    python task1.py --eda-only    # only the EDA
    python task1.py --small       # use ratings_small.csv while developing
    python task1.py --no-ratings  # EDA without the (slow) ratings passes
"""
import argparse
from collections import Counter
from pathlib import Path

from src import clean, insert
from src.DbConnector import DbConnector
from src.eda import run_eda
from src.paths import DATA_DIR


def clean_and_insert(data_dir, small):
    connection = DbConnector()
    db = connection.db
    try:
        print("\n=== Cleaning movies ===")
        movie_docs = clean.clean_movies(data_dir)

        print("\n=== Inserting movies ===")
        insert.reset_collections(db)
        insert.insert_movies(db, movie_docs)
        insert.create_indexes(db)

        print("\n=== Cleaning + inserting users (streaming ratings) ===")
        link_map = clean.build_link_map(movie_docs, data_dir)
        genre_lookup = clean.build_genre_lookup(movie_docs)
        stats = Counter()
        users = clean.iter_user_documents(link_map, genre_lookup, data_dir, small, stats)
        insert.insert_users(db, users)

        print("\n=== Verifying ===")
        insert.verify(db, stats, expected_movies=clean.EXPECTED_MOVIES,
                      expected_ratings=None if small else clean.EXPECTED_RATINGS_FULL)
    finally:
        connection.close_connection()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small", action="store_true",
                        help="use ratings_small.csv (links.csv is always the full file)")
    parser.add_argument("--no-ratings", action="store_true", help="skip the ratings EDA")
    parser.add_argument("--skip-eda", action="store_true", help="only clean + insert")
    parser.add_argument("--eda-only", action="store_true", help="only run the EDA")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"folder with the CSVs (default: {DATA_DIR})")
    args = parser.parse_args()

    if not args.skip_eda:
        run_eda(args.data_dir, small=args.small, include_ratings=not args.no_ratings)
    if not args.eda_only:
        clean_and_insert(args.data_dir, args.small)


if __name__ == "__main__":
    main()
