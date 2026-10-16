"""Extra step 0 (runs first): the records split in two by a line break.

Three overviews in movies_metadata.csv contain a line break without being
quoted, so each of those movies is read as two rows (see src/repair.py).
This step runs before the rest of the EDA so that no movies are lost:

  1. prints the raw file lines around each broken record, with hidden
     characters visible (repr), which shows the cause,
  2. shows both halves column by column next to the stitched record,
  3. runs sanity checks on the stitched record (numbers, date, status, title),
  4. returns the metadata with the records repaired, for the rest of the EDA.

Finding: all three records stitch cleanly and pass every check, so they are
repaired instead of dropped.
"""
import pandas as pd

from ..repair import find_broken_rows, repair_broken_rows, stitch
from .common import section, short, show

FILENAME = "movies_metadata.csv"
MAX_CHARS = 400  # per raw line

STATUSES = {"Released", "Rumored", "Post Production", "In Production", "Planned", "Canceled", None}


def show_raw_lines(lines, needle, context=1):
    """Print the physical lines containing `needle`, plus neighbours, with repr()."""
    hits = [i for i, line in enumerate(lines) if needle in line]
    if not hits:
        print(f"  '{needle.decode()}' not found in the raw file")
        return
    i = hits[0]
    for j in range(max(i - context, 0), min(i + context + 2, len(lines))):
        text = repr(lines[j].decode("utf-8", errors="replace"))
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS // 2] + " … " + text[-MAX_CHARS // 2:]
        marker = ">>" if j in (i, i + 1) else "  "
        print(f"{marker} line {j + 1:>6}: {text}")


def sanity(stitched):
    checks = []
    for col in ("id", "budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"):
        v = stitched.get(col)
        ok = v is None or pd.isna(v) or pd.notna(pd.to_numeric(v, errors="coerce"))
        checks.append((f"{col} is a number", ok, v))
    date = pd.to_datetime(stitched.get("release_date"), errors="coerce", format="%Y-%m-%d")
    checks.append(("release_date is a date", pd.notna(date), stitched.get("release_date")))
    checks.append(("title present", pd.notna(stitched.get("title")), stitched.get("title")))
    status = stitched.get("status")
    checks.append(("status looks like a status", status in STATUSES or pd.isna(status), status))
    return checks


def run(raw, data_dir):
    """Report on the broken records and return the repaired metadata."""
    section("Broken records in movies_metadata.csv (repaired before the rest of the EDA)")
    raw = raw.reset_index(drop=True)
    columns = list(raw.columns)
    pairs, second_halves = find_broken_rows(raw)
    print(f"First halves (valid id, no title): {len(pairs)}   "
          f"Second halves (non-numeric id): {len(second_halves)}")

    lines = (data_dir / FILENAME).read_bytes().split(b"\n") if pairs else []
    for i, j in pairs:
        first = raw.loc[i]
        print(f"\n--- id {first['id']}  {first['imdb_id']}  {first['original_title']} ---")
        print("Raw file around the record (>> = the two halves):")
        show_raw_lines(lines, str(first["imdb_id"]).encode())

        if j is None:
            print(f"The row after it is not a second half - cannot stitch, will be dropped.")
            continue
        stitched, split_col, leftover = stitch(first, raw.loc[j], columns)
        print(f"\nSplit inside '{split_col}'. Column by column:")
        show([[c, short(first[c]), short(raw.loc[j, c]), short(stitched[c])] for c in columns],
             headers=["column", "first half", "second half (as read)", "stitched"])
        if leftover:
            print("WARNING - second-half fields with nowhere to go:", ", ".join(leftover))
        print("Sanity checks on the stitched record:")
        for name, ok, value in sanity(stitched):
            print(f"  [{'OK' if ok else 'FAIL'}] {name}: {short(value, 30)}")

    fixed, repaired, dropped = repair_broken_rows(raw)
    print(f"\nRepaired {repaired} records, dropped {dropped} rows that could not be repaired. "
          f"Rows: {len(raw):,} -> {len(fixed):,}")
    return fixed
