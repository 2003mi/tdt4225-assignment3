"""Extra step 4: hidden quality problems that the missing-value counts miss.

1. Same film twice  - one imdb_id under several TMDB ids.
2. Placeholders     - overview/tagline texts repeated across many movies
                      (e.g. "No overview found."): missing values in disguise.
3. Languages        - original_language "xx" (TMDB's "no language") and
                      missing, overall and among the Q9 candidates.
4. People           - the same person more than once in one movie:
                      actors with several roles (Q2, Q3, Q6, Q8), the same
                      actor twice in a movie's top 5 (Q6), ties in cast order
                      (Q6), duplicate crew entries, directors listed twice
                      (Q1, Q8), and directors who act in their own movie (Q8).

Findings:
  - No film appears under two TMDB ids.
  - 148 placeholder overviews ("No overview found." 133, "No Overview" 7,
    "No movie overview available." 3, whitespace 5) and 4 "-" taglines -> null.
  - 33 "xx" languages, 14 of them among the Q9 candidates -> handled in Q9.
  - 951 movies list an actor several times (1,207 extra entries), 151 of them
    with the same actor twice in the top 5 -> one cast entry per person with
    all characters in a list. 87 movies have ties in cast order -> sort by
    order with the original position as tie-breaker.
  - Crew has no duplicate entries and no director is listed twice.
  - 3,754 movies have a director who also acts in them -> exclude
    director == actor pairs in Q8.
"""
from collections import Counter

import pandas as pd

from .common import pct, section, short, show, subsection

EXAMPLES = 10


def check_same_film(meta: pd.DataFrame) -> None:
    subsection("Same imdb_id under different TMDB ids")
    m = meta.drop_duplicates("id")
    m = m[m["imdb_id"].notna()]
    dup = m[m["imdb_id"].duplicated(keep=False)].sort_values(["imdb_id", "id"])
    print(f"{dup['imdb_id'].nunique()} imdb ids used by more than one TMDB id ({len(dup)} movies)")
    if len(dup):
        show([[r.imdb_id, r.id, short(r.title, 40), r.vote_count]
              for r in dup.head(2 * EXAMPLES).itertuples()],
             headers=["imdb_id", "id", "title", "vote_count"])


def check_placeholders(meta: pd.DataFrame) -> None:
    m = meta.drop_duplicates("id")
    for col in ("overview", "tagline"):
        text = m[col].dropna().str.strip()
        counts = text.value_counts()
        repeated = counts[counts >= 3]
        subsection(f"{col}: {len(repeated)} texts used by >= 3 movies "
                   f"({int(repeated.sum()):,} movies in total)")
        if len(repeated):
            show([[short(t, 60) if t else "(only whitespace)", c] for t, c in repeated.head(15).items()],
                 headers=["text", "movies"])
    print("Generic texts like 'No overview found.' are missing values in disguise;\n"
          "real taglines can repeat too, so judge from the text.")


def check_languages(meta: pd.DataFrame) -> None:
    subsection("original_language")
    m = meta.drop_duplicates("id")
    lang = m["original_language"]
    print(f"Missing: {int(lang.isna().sum())}   'xx' (no language): {int((lang == 'xx').sum())}")
    has_us = m["production_countries"].map(
        lambda lst: isinstance(lst, list) and any(c.get("iso_3166_1") == "US" for c in lst))
    q9 = m[lang.notna() & (lang != "en") & has_us]
    print(f"Q9 candidates (original_language != 'en', US production country): {len(q9):,}, "
          f"of which 'xx': {int((q9['original_language'] == 'xx').sum())}")


def check_people(credits: pd.DataFrame, titles: pd.Series) -> None:
    """credits: one row per movie with cast/crew parsed (from core.eda_credits)."""
    n = len(credits)
    stats: Counter[str] = Counter()
    repeated_actors, directors_acting = [], []

    for movie_id, cast, crew in zip(credits["id"], credits["cast"], credits["crew"]):
        cast, crew = cast or [], crew or []
        title = short(titles.get(movie_id, ""), 35)

        # Actors with several cast entries
        per_actor = Counter(p.get("id") for p in cast)
        repeats = {pid: c for pid, c in per_actor.items() if c > 1}
        if repeats:
            stats["cast_movies"] += 1
            stats["cast_extra"] += sum(c - 1 for c in repeats.values())
            for pid, c in repeats.items():
                entries = [p for p in cast if p.get("id") == pid]
                repeated_actors.append([movie_id, title, short(entries[0].get("name"), 25), c,
                                        short(" / ".join(str(p.get("character")) for p in entries), 45)])

        # Q6: top 5 by order, and ties in order
        top5 = sorted(cast, key=lambda p: p.get("order", float("inf")))[:5]
        if len({p.get("id") for p in top5}) < len(top5):
            stats["top5_dup"] += 1
        orders = [p.get("order") for p in cast]
        if len(set(orders)) < len(orders):
            stats["order_ties"] += 1

        # Crew: exact (person, job) duplicates, directors listed twice
        per_job = Counter((p.get("id"), p.get("job")) for p in crew)
        extra = sum(c - 1 for c in per_job.values() if c > 1)
        if extra:
            stats["crew_movies"] += 1
            stats["crew_extra"] += extra
        if any(c > 1 for (_, job), c in per_job.items() if job == "Director"):
            stats["director_twice"] += 1
        jobs_per_person = Counter(pid for pid, _ in per_job)
        stats["multi_job_people"] += sum(1 for c in jobs_per_person.values() if c > 1)

        # Q8: director who is also in the cast
        directors = {p.get("id"): p.get("name") for p in crew if p.get("job") == "Director"}
        acting = set(directors) & set(per_actor)
        if acting:
            stats["director_acts"] += 1
            for pid in acting:
                directors_acting.append([movie_id, title, short(directors[pid], 30)])

    subsection("Cast: the same person more than once in a movie")
    print(f"Movies where an actor has several cast entries: {stats['cast_movies']:,} "
          f"({pct(stats['cast_movies'], n)}), {stats['cast_extra']:,} extra entries")
    print(f"Movies where the top 5 by order contain the same actor twice (Q6): {stats['top5_dup']:,}")
    print(f"Movies with ties in cast order: {stats['order_ties']:,}")
    if repeated_actors:
        repeated_actors.sort(key=lambda r: -r[3])
        show(repeated_actors[:EXAMPLES], headers=["id", "title", "actor", "entries", "characters"])

    subsection("Crew")
    print(f"Movies with exact duplicate (person, job) entries: {stats['crew_movies']:,}, "
          f"{stats['crew_extra']:,} extra entries")
    print(f"Movies where the same person is listed as Director twice: {stats['director_twice']:,}")
    print(f"(person, movie) combinations with several different jobs: {stats['multi_job_people']:,} "
          f"(normal, e.g. Director + Writer)")

    subsection("Directors who also act in the same movie (Q8)")
    print(f"Movies: {stats['director_acts']:,}")
    if directors_acting:
        show(directors_acting[:EXAMPLES], headers=["id", "title", "director"])


def run(meta: pd.DataFrame, credits: pd.DataFrame, titles: pd.Series) -> None:
    """meta: from core.eda_metadata. credits: from core.eda_credits."""
    section("Extra: hidden quality problems")
    check_same_film(meta)
    check_placeholders(meta)
    check_languages(meta)
    check_people(credits, titles)
