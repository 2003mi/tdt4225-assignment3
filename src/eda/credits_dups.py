"""Extra step 3: credits.csv rows that share an id but are not exact copies.

The main EDA found 43 duplicated ids in credits.csv, 37 of them exact copies.
For the rest, this step shows how the copies differ: cast/crew size per copy,
and which entries exist in only one copy. Entries are compared by
(person id, character) for cast and (person id, job) for crew, so copies with
the same people in a different order are reported as "same entries,
different order".

Finding: all 7 differing ids have an identical cast and only a differently
ordered crew, so it does not matter which copy is kept.
"""
from ..load import parse_literal, to_number
from .common import section, short, show

MAX_LINES = 10  # entries printed per difference


def entry_key(e, kind):
    return (e.get("id"), e.get("character") if kind == "cast" else e.get("job"))


def describe(e, kind):
    if kind == "cast":
        return f"{e.get('name')} as {e.get('character')!r} (order {e.get('order')})"
    return f"{e.get('name')} - {e.get('job')} ({e.get('department')})"


def compare(copies, kind):
    """Print how the cast or crew lists differ between the copies of one movie."""
    lists = [c or [] for c in copies]
    keys = [[entry_key(e, kind) for e in lst] for lst in lists]
    if all(k == keys[0] for k in keys):
        print(f"  {kind}: identical")
        return "identical"
    if all(sorted(map(str, k)) == sorted(map(str, keys[0])) for k in keys):
        print(f"  {kind}: same entries, different order")
        return "order"
    for i, lst in enumerate(lists):
        others = set().union(*(set(k) for j, k in enumerate(keys) if j != i))
        only_here = [e for e in lst if entry_key(e, kind) not in others]
        if only_here:
            print(f"  {kind}: {len(only_here)} entr{'y' if len(only_here) == 1 else 'ies'} "
                  f"only in copy {i + 1}:")
            for e in only_here[:MAX_LINES]:
                print(f"      + {describe(e, kind)}")
            if len(only_here) > MAX_LINES:
                print(f"      ... and {len(only_here) - MAX_LINES} more")
    return "content"


def run(raw_credits, titles):
    """raw_credits: credits.csv as text. titles: Series id -> title."""
    section("Extra: credits rows that share an id but differ")
    credits = raw_credits[to_number(raw_credits["id"]).notna()].copy()
    credits["id"] = credits["id"].astype(int)

    dups = credits[credits["id"].duplicated(keep=False)].drop_duplicates()
    differing = dups[dups["id"].duplicated(keep=False)]
    ids = sorted(int(i) for i in differing["id"].unique())
    print(f"{int(credits['id'].duplicated().sum())} duplicate rows; "
          f"{len(ids)} ids have copies that are not identical")

    summary = []
    for movie_id in ids:
        group = differing[differing["id"] == movie_id]
        cast = [parse_literal(v, list) for v in group["cast"]]
        crew = [parse_literal(v, list) for v in group["crew"]]
        title = titles.get(movie_id, "(not in metadata)")
        print(f"\nid {movie_id}: {title} ({len(group)} distinct copies)")
        show([[i + 1, len(c or []), len(w or [])] for i, (c, w) in enumerate(zip(cast, crew))],
             headers=["copy", "cast", "crew"])
        summary.append([movie_id, short(title, 40), compare(cast, "cast"), compare(crew, "crew")])

    if summary:
        print("\nSummary")
        show(summary, headers=["id", "title", "cast", "crew"])
