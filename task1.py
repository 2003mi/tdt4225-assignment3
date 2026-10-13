"""Part 1: EDA, cleaning and inserting the Movies dataset.

    python task1.py               # EDA on the full dataset
    python task1.py --small       # use ratings_small.csv / links_small.csv while developing
    python task1.py --no-ratings  # skip the (slow) pass over ratings.csv
"""
import argparse
from pathlib import Path

from src.eda import run_eda
from src.paths import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small", action="store_true",
                        help="use ratings_small.csv and links_small.csv")
    parser.add_argument("--no-ratings", action="store_true", help="skip the ratings EDA")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"folder with the CSVs (default: {DATA_DIR})")
    args = parser.parse_args()

    run_eda(args.data_dir, small=args.small, include_ratings=not args.no_ratings)

    # TODO: clean + insert into MongoDB


if __name__ == "__main__":
    main()