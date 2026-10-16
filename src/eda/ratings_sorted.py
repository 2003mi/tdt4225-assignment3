"""Extra step 5: is the ratings file sorted and grouped by userId?

Reads only the userId column, in chunks, and reports
  - whether userId never decreases from one row to the next (sorted),
  - whether each user's ratings form one contiguous block (grouped). A file
    can be unsorted but still grouped, which is enough to build one document
    per user while streaming,
  - the first place the order breaks, as a line number in the CSV.

Finding: ratings.csv (26,024,289 rows, 270,896 users) is both sorted and
grouped by userId. The chunked duplicate check in the main EDA relies on this,
and so does building one document per user during insert.
"""
import time

import numpy as np
import pandas as pd

from .common import section


def check(path, chunksize=1_000_000):
    rows = nulls = descending = reappearing = 0
    first_descending = first_reappearing = None
    seen = set()   # users whose block has started
    prev = None    # last userId of the previous chunk
    start = time.time()

    for chunk in pd.read_csv(path, usecols=["userId"], chunksize=chunksize):
        rows += len(chunk)
        nulls += int(chunk["userId"].isna().sum())
        vals = chunk["userId"].to_numpy()
        lines = chunk.index.to_numpy() + 2  # +1 for the header, +1 for 1-based lines

        # Prepend the last value of the previous chunk so the boundary is checked too.
        full = vals if prev is None else np.concatenate([[prev], vals])
        offset = 1 if prev is None else 0   # full[k] is this chunk's row k - (1 - offset)
        diffs = np.diff(full)

        # Sorted: userId must never go down
        down = np.flatnonzero(diffs < 0)
        descending += len(down)
        if len(down) and first_descending is None:
            j = down[0]  # full[j] -> full[j + 1] goes down
            first_descending = (int(lines[j + offset]), full[j], full[j + 1])

        # Grouped: every change of userId starts a block; a block for a user
        # already seen means that user's ratings are split up.
        block_starts = np.flatnonzero(diffs != 0) + 1
        if prev is None:
            block_starts = np.concatenate([[0], block_starts])
        for k in block_starts:
            user = full[k]
            if user in seen:
                reappearing += 1
                if first_reappearing is None:
                    first_reappearing = (int(lines[k - (1 - offset)]), user)
            seen.add(user)

        prev = full[-1]
        print(f"  checked {rows:,} rows...", end="\r")

    print(" " * 40, end="\r")
    return {
        "rows": rows, "nulls": nulls, "users": len(seen),
        "descending": descending, "first_descending": first_descending,
        "reappearing": reappearing, "first_reappearing": first_reappearing,
        "seconds": time.time() - start,
    }


def run(data_dir, small=False):
    name = "ratings_small.csv" if small else "ratings.csv"
    section(f"Extra: is {name} sorted by userId?")
    r = check(data_dir / name)
    print(f"Rows: {r['rows']:,}   Users: {r['users']:,}   Empty userId: {r['nulls']:,}   "
          f"({r['seconds']:.1f} s)")

    if r["descending"] == 0:
        print("Sorted by userId: YES")
    else:
        line, before, after = r["first_descending"]
        print(f"Sorted by userId: NO - userId goes down {r['descending']:,} times "
              f"(first at line {line:,}: {before} -> {after})")

    if r["reappearing"] == 0:
        print("Each user's ratings are in one contiguous block: YES")
    else:
        line, user = r["first_reappearing"]
        print(f"Each user's ratings are in one contiguous block: NO - {r['reappearing']:,} "
              f"users reappear later (first: user {user} at line {line:,})")
