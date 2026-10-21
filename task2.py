"""Part 2: run the queries and print the results.

    python task2.py              # all questions
    python task2.py 1 7          # only Q1 and Q7
    python task2.py 10           # Q10 (both parts: 10a and 10b)
    python task2.py 2 --rows 50  # print more rows of a long result
    python task2.py 7 --pprint   # also pprint the raw documents from MongoDB
    python task2.py 7 --explain  # also print how MongoDB ran it (index used, docs read, time)

Each question is a function in src/queries.py. Results are printed as a table
(screenshot these for the report); --pprint additionally shows the raw rows
with pprint, as the assignment suggests.
"""
import argparse
import time
from datetime import datetime
from pprint import pprint
from typing import Any

from tabulate import tabulate

from src import queries
from src.DbConnector import DbConnector
from src.queries import QUERIES, Row


def format_value(value: Any) -> Any:
    """Make one cell readable: dates as YYYY-MM-DD, lists joined with commas,
    floats rounded; everything else unchanged."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, int) and not isinstance(value, bool):
        # pymongo returns large integers as bson.Int64, which tabulate would
        # print as a float ("7,707,367,425.00"); a plain int prints as "7,707,367,425"
        return int(value)
    return value


def print_rows(rows: list[Row], max_rows: int) -> None:
    """Print the rows as a table; the columns come from the first row's keys."""
    if not rows:
        print("(no rows)")
        return
    columns = list(rows[0].keys())
    table = [[format_value(row.get(col)) for col in columns] for row in rows[:max_rows]]
    # Text columns are printed as-is: tabulate would otherwise read text like
    # "45811" or "1960" as a number and crash applying intfmt="," to it.
    text_columns = [i for i in range(len(columns)) if any(isinstance(r[i], str) for r in table)]
    print(tabulate(table, headers=columns, tablefmt="simple", floatfmt=",.2f", intfmt=",",
                   disable_numparse=text_columns))
    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows:,} more rows ({len(rows):,} in total)")


def selected_keys(requested: list[str]) -> list[str]:
    """'10' selects 10a and 10b; no arguments selects everything."""
    if not requested:
        return list(QUERIES)
    keys = [k for r in requested for k in QUERIES if k == r or k.rstrip("ab") == r]
    unknown = [r for r in requested if not any(k == r or k.rstrip("ab") == r for k in QUERIES)]
    if unknown:
        raise SystemExit(f"Unknown question(s): {', '.join(unknown)}. Choose from {', '.join(QUERIES)}")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions", nargs="*", help="question numbers, e.g. 1 7 10 (default: all)")
    parser.add_argument("--rows", type=int, default=25, help="max rows to print per result (default 25)")
    parser.add_argument("--pprint", action="store_true", help="also pprint the raw result rows")
    parser.add_argument("--explain", action="store_true",
                        help="print MongoDB's execution plan for every aggregation (runs each query twice)")
    args = parser.parse_args()
    keys = selected_keys(args.questions)
    queries.EXPLAIN = args.explain

    connection = DbConnector()
    try:
        db = connection.db
        for key in keys:
            title, query = QUERIES[key]
            print("\n" + "=" * 72)
            print(f"Q{key}: {title}")
            print("=" * 72)
            start = time.perf_counter()
            try:
                rows = query(db)
            except NotImplementedError:
                print("(not implemented yet)")
                continue
            print_rows(rows, args.rows)
            if args.pprint:
                pprint(rows[: args.rows], sort_dicts=False)
            print(f"({time.perf_counter() - start:.1f} s)")
    finally:
        connection.close_connection()


if __name__ == "__main__":
    main()