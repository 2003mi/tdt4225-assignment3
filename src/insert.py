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

Status: functions marked TODO are skeletons to fill in. The rest is done.
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
           expected_ratings: int | None = None) -> None:
    """TODO: check the database against the expected numbers and print evidence for the report.

    - movies: count_documents({}) == expected_movies
    - users: count_documents({}) == stats["users"]
    - total ratings in the database == stats["ratings_kept"] (and ==
      expected_ratings on the full file). Hint: aggregate
      [{"$group": {"_id": None, "n": {"$sum": {"$size": "$ratings"}}}}]
    - print stats (read / unmapped / kept)
    - pprint one movie (e.g. Toy Story, _id 862) and the first rating of one user
    - largest user document in bytes:
      [{"$project": {"size": {"$bsonSize": "$$ROOT"}}}, {"$sort": {"size": -1}}, {"$limit": 1}]
    - type check: one movie's release_date is a datetime, budget an int or None
    """
    raise NotImplementedError
