"""Insert the cleaned documents into MongoDB and verify the result.

Collections
-----------
movies - one document per movie, _id = TMDB id. Credits (cast grouped per
         person, crew trimmed) and keywords are embedded: they are 1:1 with
         the movie, always read together with it in the queries, and bounded
         in size (largest cast 313, crew 435).
users  - one document per user, _id = MovieLens userId, with an embedded
         "ratings" array. Each rating holds the TMDB movie id and a copy of the
         movie's genres (denormalised for Q10: written once, read many times,
         so the copies never go out of sync). The largest user has 18,276
         ratings, far below the 16 MB document limit.

MongoDB needs no schema up front: collections are created on the first insert.
Both collections are dropped first, so the script can be rerun safely.

"""
from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import islice
from pprint import pprint
from typing import TypeVar

from pymongo.database import Database

from .schema import MovieDoc, UserDoc

T = TypeVar("T")

BATCH_SIZE = 5_000  # documents per insert_many call


def reset_collections(db: Database) -> None:
    """Drop movies and users so a rerun starts from scratch."""
    for name in ("movies", "users"):
        db[name].drop()
    print("Dropped collections: movies, users")


def batched(iterable: Iterable[T], size: int = BATCH_SIZE) -> Iterator[list[T]]:
    """Yield lists of up to `size` items (works on generators, too)."""
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


def insert_movies(db: Database, movie_docs: list[MovieDoc]) -> None:
    for batch in batched(movie_docs):
        db.movies.insert_many(batch, ordered=False)
    print(f"Inserted {db.movies.count_documents({}):,} movies")


def create_indexes(db: Database) -> None:
    """Indexes are created after the bulk insert (faster than updating them per insert).

    _id is always indexed. The text index is required for Q7.
    """
    db.movies.create_index([("overview", "text"), ("tagline", "text")], name="text_overview_tagline")
    print("Created text index on movies.overview + movies.tagline")


def insert_users(db: Database, user_docs: Iterable[UserDoc]) -> None:
    """Insert user documents from a generator, in batches, printing progress."""
    inserted = 0
    for batch in batched(user_docs):
        db.users.insert_many(batch, ordered=False)
        inserted += len(batch)
        print(f"  inserted {inserted:,} users...", end="\r")
    print(f"Inserted {inserted:,} users" + " " * 20)


def verify(db: Database, stats: Counter[str], expected_movies: int,
           expected_ratings: int | None = None) -> bool:
    """Check the database against the expected numbers and print evidence for the report.

    Counts   - movies, users and the total number of embedded ratings match
               what was built (and the EDA's numbers on the full file)
    Size     - the largest user document is far below MongoDB's 16 MB limit
    Types    - dates are real dates, numbers are numbers, no text "None"
    Index    - the text index for Q7 exists
    Samples  - one movie and one user, shortened with $slice

    Prints [OK] / [FAIL] per check and returns True if everything passed.
    The ratings check reads the whole users collection once (can take a minute).
    """
    results: list[bool] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")

    # --- Counts and size
    print("Counts")
    n_movies = db.movies.count_documents({})
    check("movies", n_movies == expected_movies, f"{n_movies:,} (expected {expected_movies:,})")
    n_users = db.users.count_documents({})
    check("users", n_users == stats["users"], f"{n_users:,} (built {stats['users']:,})")

    totals = next(db.users.aggregate([
        {"$project": {"n": {"$size": "$ratings"}, "bytes": {"$bsonSize": "$$ROOT"}}},
        {"$group": {"_id": None, "ratings": {"$sum": "$n"},
                    "max_ratings": {"$max": "$n"}, "max_bytes": {"$max": "$bytes"}}},
    ]), {"ratings": 0, "max_ratings": 0, "max_bytes": 0})
    check("ratings in users", totals["ratings"] == stats["ratings_kept"],
          f"{totals['ratings']:,} (built {stats['ratings_kept']:,})")
    if expected_ratings is not None:
        check("ratings vs. EDA", totals["ratings"] == expected_ratings,
              f"{totals['ratings']:,} (expected {expected_ratings:,})")
    print(f"  ratings read {stats['ratings_read']:,}, unmapped (no valid link) "
          f"{stats['ratings_unmapped']:,}, kept {stats['ratings_kept']:,}")
    limit = 16 * 1024 * 1024
    check("largest user document", totals["max_bytes"] < limit,
          f"{totals['max_ratings']:,} ratings, {totals['max_bytes'] / 1024 / 1024:.2f} MB (limit 16 MB)")

    # --- Types: every value has the type the queries expect
    print("Types")
    def not_of_type(field: str, *types: str) -> int:
        """Movies where `field` is none of `types` and not null."""
        return db.movies.count_documents(
            {"$nor": [{field: {"$type": t}} for t in types] + [{field: None}]})

    wrong = not_of_type("release_date", "date")
    check("movies.release_date is a date or null", wrong == 0, f"{wrong:,} wrong")
    for field in ("budget", "revenue", "runtime", "vote_count"):
        wrong = not_of_type(field, "int", "long")
        check(f"movies.{field} is an integer or null", wrong == 0, f"{wrong:,} wrong")
    wrong = not_of_type("vote_average", "double")
    check("movies.vote_average is a double or null", wrong == 0, f"{wrong:,} wrong")
    wrong = db.movies.count_documents({"belongs_to_collection": {"$type": "string"}})
    check("no collection stored as text", wrong == 0, f"{wrong:,} wrong")
    wrong = next(db.users.aggregate([
        {"$sample": {"size": 1000}},
        {"$unwind": "$ratings"},
        {"$match": {"ratings.timestamp": {"$not": {"$type": "date"}}}},
        {"$count": "n"},
    ]), {"n": 0})["n"]
    check("users.ratings.timestamp is a date (1,000 random users)", wrong == 0, f"{wrong:,} wrong")

    # --- Index
    print("Index")
    indexes = [ix["name"] for ix in db.movies.list_indexes()]
    check("text index for Q7", "text_overview_tagline" in indexes, ", ".join(indexes))

    # --- Samples for the report
    short_lists = {"cast": {"$slice": 3}, "crew": {"$slice": 3}, "keywords": {"$slice": 3}}
    movie = db.movies.find_one({"_id": 862}, short_lists) or db.movies.find_one({}, short_lists)
    print("Sample movie (Toy Story, _id 862, if present; lists cut to 3 with $slice)")
    pprint(movie, sort_dicts=False)
    print("Sample user (lowest _id; first 2 ratings)")
    pprint(db.users.find_one({}, {"ratings": {"$slice": 2}}, sort=[("_id", 1)]), sort_dicts=False)

    passed = sum(results)
    print(f"\nVerification: {passed}/{len(results)} checks passed"
          + ("" if passed == len(results) else " - see [FAIL] above"))
    return passed == len(results)
