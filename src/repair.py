"""Repair the records in movies_metadata.csv that a line break split in two.

Three overviews contain a line break but are not wrapped in quotes in the CSV.
A line break outside quotes ends a CSV record, so each of these movies is read
as two rows:
  - a "first half": valid id, filled up to (part of) the overview, rest empty,
  - a "second half": starts with the rest of the overview, and every following
    field is shifted left, so e.g. the release date lands in the id column.

stitch() puts the two halves back together. Used by the EDA (which reports
the repair) and by the cleaning step.
"""
import pandas as pd

from .load import to_number


def find_broken_rows(raw):
    """Locate the halves. `raw` must have a 0..n-1 index (reset_index).

    Returns (pairs, second_halves): pairs is a list of (first_index,
    second_index or None), second_halves the set of rows with a non-numeric id.
    """
    ids = to_number(raw["id"])
    first_halves = list(raw.index[ids.notna() & raw["title"].isna()])
    second_halves = set(raw.index[ids.isna()])
    pairs = [(i, i + 1 if i + 1 in second_halves else None) for i in first_halves]
    return pairs, second_halves


def stitch(first, second, columns):
    """Join a first and second half into one record.

    The first half is kept up to its last filled column (where the split is);
    the second half's first field is the rest of that column, and its other
    fields follow in order. Returns (record dict, split column, leftover),
    where leftover lists second-half fields that had no column to go to -
    if it is not empty, the split was not a clean shift.
    """
    filled = [c for c in columns if pd.notna(first[c])]
    k = columns.index(filled[-1])
    stitched = {c: first[c] for c in columns[:k]}
    tail = second[columns[0]]
    stitched[columns[k]] = first[columns[k]] if pd.isna(tail) else f"{first[columns[k]]} {str(tail).strip()}"
    leftover = []
    for j, c in enumerate(columns[1:], start=1):
        target = k + j
        if target < len(columns):
            stitched[columns[target]] = second[c]
        elif pd.notna(second[c]):
            leftover.append(c)
    for c in columns:
        stitched.setdefault(c, None)
    return stitched, columns[k], leftover


def repair_broken_rows(raw):
    """Return (repaired DataFrame, number of records repaired, number dropped).

    Halves that cannot be stitched cleanly are dropped instead.
    """
    raw = raw.reset_index(drop=True)
    columns = list(raw.columns)
    pairs, second_halves = find_broken_rows(raw)

    fixed = raw.copy()
    repaired, unrepaired = 0, []
    for i, j in pairs:
        if j is not None:
            stitched, _, leftover = stitch(raw.loc[i], raw.loc[j], columns)
            if not leftover:
                fixed.loc[i] = [stitched[c] for c in columns]
                repaired += 1
                continue
        unrepaired.append(i)

    drop = sorted(second_halves | set(unrepaired))
    fixed = fixed.drop(index=drop).reset_index(drop=True)
    return fixed, repaired, len(drop) - repaired
