"""Part 2: the ten questions, one function each.

Every query function takes the database and returns a list of result rows
(dicts). task2.py runs them, times them and prints the rows as a table.
A function that still raises NotImplementedError is reported as "not
implemented yet" and the other queries still run.

Data model (see src/schema.py for every field):
  movies - one document per movie, _id = TMDB id. Embedded: genres (order kept,
           genres[0] = primary genre), belongs_to_collection {id, name} or null,
           production_countries [{iso_3166_1, name}], keywords,
           cast [{id, name, gender, order, characters}] - one entry per person,
           sorted by order - and crew [{id, name, job, department}].
           Unknown budget/revenue/runtime/vote_average are null, release_date
           is a date or null. Text index on overview + tagline.
  users  - one document per user, _id = MovieLens userId, ratings
           [{movie_id, rating, timestamp, genres [{id, name}]}].

Rules from Part 1 that the queries rely on:
  - aggregations like $median / $avg / $sum ignore null, so unknown values
    drop out by themselves - but $count and $sum: 1 still count the movie
  - group people by id, not by name (different people can share a name)
  - the cast has one entry per person per movie, so a person is never counted
    twice in one movie

Syntax reminders used in the comments below:
  "$field"       the value of a field in the current document
  "a.b"          field b inside subdocument a; on an array of subdocuments,
                 "$a.b" gives the array of all the b values
  "$$name"       a variable, e.g. the loop variable of $map / $filter / $reduce
  _id in $group  the group key (like SQL GROUP BY); every other field in a
                 $group must be an accumulator ($sum, $avg, $first, ...)

aggregate() always passes allowDiskUse=True, so the heavy pipelines (Q2, Q8,
Q10) may spill to disk instead of failing at MongoDB's 100 MB memory limit.
"""
from typing import Any

from pymongo.collection import Collection
from pymongo.database import Database
from pymongoexplain import ExplainableCollection

Row = dict[str, Any]

# task2.py --explain sets this to True: every aggregate() call then also
# prints how MongoDB executed the pipeline.
EXPLAIN = False


def aggregate(collection: Collection, pipeline: list[dict[str, Any]],
              explain: bool = False) -> list[Row]:
    """Run an aggregation pipeline and return all result documents as a list.

    allowDiskUse lets MongoDB spill large intermediate results (e.g. all actor
    pairs) to disk instead of failing at its 100 MB in-memory limit.

    explain=True (or EXPLAIN, set by task2.py --explain) first prints a short
    summary of MongoDB's execution plan, then runs the query as usual, so the
    function still returns the rows.
    """
    if explain or EXPLAIN:
        explain_pipeline(collection, pipeline)
    return list(collection.aggregate(pipeline, allowDiskUse=True))


def explain_pipeline(collection: Collection, pipeline: list[dict[str, Any]],
                     verbosity: str = "executionStats") -> dict[str, Any]:
    """Ask MongoDB how it runs `pipeline`, print a summary and return the full plan.

    ExplainableCollection(...).aggregate() sends an "explain" command and
    returns its result as ONE dict (the plan), not a cursor of rows.
    verbosity: "queryPlanner" only plans the query (fast); "executionStats"
    also runs it and reports documents examined and time (so the query runs
    twice in total with --explain).
    Options must be given in camelCase (allowDiskUse): pymongoexplain silently
    drops snake_case ones such as allow_disk_use.
    """
    plan: dict[str, Any] = ExplainableCollection(collection, verbosity=verbosity).aggregate(
        pipeline, allowDiskUse=True)
    print_plan_summary(plan, verbosity)
    return plan


def _find(obj: Any, key: str) -> list[Any]:
    """All values stored under `key` anywhere inside nested dicts/lists, top-down."""
    found: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(_find(v, key))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find(item, key))
    return found


def print_plan_summary(plan: dict[str, Any], verbosity: str) -> None:
    """Print the parts of an explain result that matter for the report.

    pipeline stages - the stages MongoDB runs after the query layer ("$cursor"
                      is the query layer; stages pushed into it, e.g. a first
                      $match, $group or $sort, no longer appear on their own)
    plan            - the winning plan, outermost stage first. COLLSCAN = every
                      document read; IXSCAN / TEXT_MATCH = an index was used
    executionStats  - documents/keys examined, rows returned and time
    """
    print(f"  explain ({verbosity}):")
    stages = [next(iter(stage)) for stage in plan.get("stages", []) if stage]
    if stages:
        print(f"    pipeline stages: {' -> '.join(stages)}")
    winning = _find(plan, "winningPlan")
    plan_stages = _find(winning[0], "stage") if winning else []
    indexes = sorted(set(_find(winning, "indexName")))
    if plan_stages:
        used = f"  (index: {', '.join(indexes)})" if indexes else "  (no index)"
        print(f"    plan: {' <- '.join(plan_stages)}{used}")
    stats = next((s for s in _find(plan, "executionStats") if isinstance(s, dict)), None)
    if stats:
        print(f"    docs examined {stats.get('totalDocsExamined', 0):,}, "
              f"keys examined {stats.get('totalKeysExamined', 0):,}, "
              f"returned {stats.get('nReturned', 0):,}, "
              f"time {stats.get('executionTimeMillis', 0):,} ms")


# =============================================================================
# The ten questions
# =============================================================================

def q1(db: Database) -> list[Row]:
    """Considering only crew with job = Director, which 10 directors with >= 5
    movies have the highest median revenue? Also report each director's movie
    count and mean vote_average.

    Decided: a director needs >= 5 directed movies in total (also movies with
    unknown revenue); the median revenue is taken over the movies whose revenue
    is known. A movie with several directors counts for each of them.
    Columns: director, movies, median_revenue, mean_vote_average.
    """
    return aggregate(db.movies, [
        # Keep only movies with at least one director. 
        {"$match": {"crew.job": "Director"}},
        # One document per crew member.
        {"$unwind": "$crew"},
        # Same filter again, but now "crew" is a single person, so this keeps
        # exactly the director rows: one document per (movie, director) pair.
        {"$match": {"crew.job": "Director"}},
        # One document per director, grouped by person id (names can repeat).
        #   $sum: 1   counts the documents = movies directed (unknown revenue too)
        #   $median / $avg skip null, so unknown revenues/scores drop out
        {"$group": {
            "_id": "$crew.id",
            "director": {"$first": "$crew.name"},
            "movies": {"$sum": 1},
            "median_revenue": {"$median": {"input": "$revenue", "method": "approximate"}},
            "mean_vote_average": {"$avg": "$vote_average"},
        }},
        # $match after $group works like SQL HAVING: directors with >= 5 movies
        {"$match": {"movies": {"$gte": 5}}},
        # Highest median revenue first; the director id breaks ties so the
        # order is the same on every run
        {"$sort": {"median_revenue": -1, "_id": 1}},
        {"$limit": 10},
        # Output columns: hide the numeric _id (1 = keep, 0 = drop)
        {"$project": {"_id": 0, "director": 1, "movies": 1,
                      "median_revenue": 1, "mean_vote_average": 1}},
    ])


def q2(db: Database) -> list[Row]:
    """List actor pairs that have co-starred in >= 3 movies together, their
    number of co-appearances, and their average movie vote_average. Sort by
    number of co-appearances.

    Notes: count each pair once - (A, B) and (B, A) are the same pair, so keep
    the person with the smaller id first. Big casts give many pairs (n*(n-1)/2
    per movie), so this is the heaviest query. The result can be long: return
    it all, task2.py prints the first rows and the total.
    Columns: actor_1, actor_2, movies, avg_vote_average.
    """

    cast_ids_names = {"$map": {"input": "$cast", "as": "c",
                               "in": {"id": "$$c.id", "name": "$$c.name"}}}
    return aggregate(db.movies, [
        # Two copies of the (small) cast list.
        {"$project": {"vote_average": 1, "a": cast_ids_names, "b": cast_ids_names}},
        # Unwinding both copies gives every actor x actor combination inside a
        # movie: a cast of 3 gives 3 x 3 = 9 documents
        {"$unwind": "$a"},
        {"$unwind": "$b"},
        # Keep each pair once, with the smaller id first. This removes the
        # actor-with-themselves pairs (5, 5) and the mirrored duplicates
        # ((9, 5) when (5, 9) is kept)..
        {"$match": {"$expr": {"$lt": ["$a.id", "$b.id"]}}},
        # One document per pair: the compound _id groups on both ids at once
        {"$group": {
            "_id": {"a": "$a.id", "b": "$b.id"},
            "actor_1": {"$first": "$a.name"},
            "actor_2": {"$first": "$b.name"},
            "movies": {"$sum": 1},
            "avg_vote_average": {"$avg": "$vote_average"},
        }},
        # HAVING movies >= 3 (before the sort, so fewer pairs are sorted)
        {"$match": {"movies": {"$gte": 3}}},
        # Most co-appearances first; ties: higher score first, then the ids
        {"$sort": {"movies": -1, "avg_vote_average": -1, "_id": 1}},
        {"$project": {"_id": 0, "actor_1": 1, "actor_2": 1, "movies": 1, "avg_vote_average": 1}},
    ])


def q3(db: Database) -> list[Row]:
    """List the top 10 actors (with >= 10 credited movies) that have the widest
    genre breadth. Report the actor, the number of distinct genres they have
    appeared in, and up to 5 example genres.

    Columns: actor, movies, distinct_genres, example_genres.
    """
    return aggregate(db.movies, [
        # One document per (movie, actor).
        {"$unwind": "$cast"},
        {"$group": {
            "_id": "$cast.id",
            "actor": {"$first": "$cast.name"},
            "movies": {"$sum": 1},
            "genres": {"$addToSet": "$genres.name"},
        }},
        # >= 10 credited movies (before the $reduce, so it runs for fewer actors)
        {"$match": {"movies": {"$gte": 10}}},
        # Flatten the lists into one list of distinct genres. $reduce is a loop
        # with a running result: $$value = result so far (starts as []),
        # $$this = the current movie's genre list, $setUnion merges the two
        # and removes duplicates.
        {"$addFields": {"genres": {"$reduce": {
            "input": "$genres",
            "initialValue": [],
            "in": {"$setUnion": ["$$value", "$$this"]},
        }}}},
        # Separate stage: a stage cannot use a field it creates itself
        {"$addFields": {"distinct_genres": {"$size": "$genres"}}},
        # Check used for the report (how many actors cover all 20 genres);
        # not part of the answer:
        # {"$match": {"distinct_genres": 20}},
        # {"$count": "actors_with_all_genres"},
        # Widest breadth first; ties: more movies first, then actor id
        {"$sort": {"distinct_genres": -1, "movies": -1, "_id": 1}},
        {"$limit": 10},
        # $project can compute new fields too: $slice takes the first 5 genres
        {"$project": {"_id": 0, "actor": 1, "movies": 1, "distinct_genres": 1,
                      "example_genres": {"$slice": ["$genres", 5]}}},
    ])


def q4(db: Database) -> list[Row]:
    """For film collections (belongs_to_collection.name not null) with >= 3
    movies, which 10 collections have the largest total revenue? For each,
    report movie count, total revenue, median vote_average, and the earliest
    -> latest release date.

    Columns: collection, movies, total_revenue, median_vote_average, release_span.
    """
    return aggregate(db.movies, [
        # Only movies in a collection. 
        {"$match": {"belongs_to_collection.name": {"$ne": None}}},
        # One document per collection (grouped by collection id).
        {"$group": {
            "_id": "$belongs_to_collection.id",
            "collection": {"$first": "$belongs_to_collection.name"},
            "movies": {"$sum": 1},
            "total_revenue": {"$sum": "$revenue"},
            "median_vote_average": {"$median": {"input": "$vote_average", "method": "approximate"}},
            "first_release": {"$min": "$release_date"},
            "last_release": {"$max": "$release_date"},
        }},
        # HAVING movies >= 3
        {"$match": {"movies": {"$gte": 3}}},
        # Largest total first; ties: more movies, then collection id
        {"$sort": {"total_revenue": -1, "movies": -1, "_id": 1}},
        {"$limit": 10},
        # "earliest -> latest" as one text column: $dateToString formats each
        # date and $concat joins the pieces
        {"$project": {"_id": 0, "collection": 1, "movies": 1, "total_revenue": 1,
                      "median_vote_average": 1, "release_span": {"$concat": [
                          {"$dateToString": {"date": "$first_release", "format": "%Y-%m-%d"}},
                          " -> ",
                          {"$dateToString": {"date": "$last_release", "format": "%Y-%m-%d"}},
                      ]}}},
    ])


def q5(db: Database) -> list[Row]:
    """By decade (1970s, 1980s, ...) and primary genre (first element in
    genres), what is the median runtime and movie count? Sort by decade, then
    median runtime (desc).

    Notes: movies without a release date or without genres cannot be placed and
    drop out; runtime null is ignored by $median. Decade = year - year % 10.
    Columns: decade, primary_genre, median_runtime, movies.
    """
    return aggregate(db.movies, [
        # Drop movies that cannot be placed. "genres.0" is the element at
        # index 0, so $exists: True means "the array is not empty".
        {"$match": {"release_date": {"$ne": None}, "genres.0": {"$exists": True}}},
        # Per-movie values ($addFields keeps all other fields):
        #   decade        = year - year % 10, e.g. 1994 - 4 = 1990
        #   primary_genre = first element of the genre-name array. Not $first:
        #                   inside $group, $first means "first DOCUMENT"
        {"$addFields": {
            "decade": {"$subtract": [{"$year": "$release_date"},
                                     {"$mod": [{"$year": "$release_date"}, 10]}]},
            "primary_genre": {"$arrayElemAt": ["$genres.name", 0]},
        }},
        # One document per (decade, primary genre): a compound group key
        {"$group": {
            "_id": {"decade": "$decade", "primary_genre": "$primary_genre"},
            "median_runtime": {"$median": {"input": "$runtime", "method": "approximate"}},
            "movies": {"$sum": 1},
        }},
        # Oldest decade first, then longest median runtime; the key fields now
        # only exist inside _id, hence "_id.decade"
        {"$sort": {"_id.decade": 1, "median_runtime": -1, "_id.primary_genre": 1}},
        # Build the output document field by field ($replaceWith keeps this
        # order; $project would put the computed fields last). The decade
        # becomes a label like "1990s" - after the sort, which needs the number.
        {"$replaceWith": {
            "decade": {"$concat": [{"$toString": "$_id.decade"}, "s"]},
            "primary_genre": "$_id.primary_genre",
            "median_runtime": "$median_runtime",
            "movies": "$movies",
        }},
    ])


def q6(db: Database) -> list[Row]:
    """For each movie's top-billed 5 cast (by order), what is the proportion of
    female cast? Aggregate by decade and list decades sorted by average female
    proportion (desc), including the movie counts used. Unknown gender may be
    ignored.

    Notes: the cast is already sorted by order, so the top 5 is the first 5
    entries. Gender: 1 = female, 2 = male, 0 = unknown (ignored). A movie whose
    top 5 has no known gender has no proportion and is not counted.
    Columns: decade, avg_female_proportion, movies.
    """
    return aggregate(db.movies, [
        {"$match": {"release_date": {"$ne": None}}},
        # Everything per movie, with array operators instead of $unwind, so
        # there stays one document per movie and $sum: 1 counts movies.
        #   "$cast.gender"  the genders in billing order, e.g. [1, 2, 2, 0, 2, 1]
        #   $slice 5        the top 5 billed (not order < 5: order can have gaps)
        #   $filter         keeps elements where cond is true, like
        #                   [g for g in top5 if g in (1, 2)] -> unknown removed
        {"$addFields": {
            "decade": {"$subtract": [{"$year": "$release_date"},
                                     {"$mod": [{"$year": "$release_date"}, 10]}]},
            "genders": {"$filter": {"input": {"$slice": ["$cast.gender", 5]}, "as": "g",
                                    "cond": {"$in": ["$$g", [1, 2]]}}},
        }},
        # A movie with no known gender in its top 5 has no proportion -> drop
        # it. This also guarantees the division below never divides by zero.
        {"$match": {"genders.0": {"$exists": True}}},
        # Proportion per movie = number of women / number of known genders
        {"$addFields": {"female_proportion": {"$divide": [
            {"$size": {"$filter": {"input": "$genders", "as": "g", "cond": {"$eq": ["$$g", 1]}}}},
            {"$size": "$genders"},
        ]}}},
        # One document per decade: the mean of the per-movie proportions (every
        # movie weighs the same), and how many movies were used
        {"$group": {
            "_id": "$decade",
            "avg_female_proportion": {"$avg": "$female_proportion"},
            "movies": {"$sum": 1},
        }},
        # Highest proportion first
        {"$sort": {"avg_female_proportion": -1, "_id": 1}},
        # Output in this column order; here _id IS the decade (not compound)
        {"$replaceWith": {
            "decade": {"$concat": [{"$toString": "$_id"}, "s"]},
            "avg_female_proportion": "$avg_female_proportion",
            "movies": "$movies",
        }},
    ])


def q7(db: Database) -> list[Row]:
    """Using a text search over overview and tagline, which 20 movies matching
    "noir" or "neo-noir" (and vote_count >= 50) have the highest vote_average?
    Return title, year, vote_average and vote_count.

    Notes: uses the text index (created in Part 1). The tokenizer splits
    "neo-noir" into "neo" and "noir", so searching "noir" finds both; a search
    string starting with "-" means "exclude". In an aggregation, a $text match
    must be the first stage.
    Result: 60 movies match "noir" (5 of them the phrase "neo-noir"), but only 5
    have >= 50 votes, so the answer has 5 rows; no neo-noir match reaches 50
    votes (max 42, The Aura).
    Columns: title, year, vote_average, vote_count.
    """
    # Checks used for the report:
    # print(db.movies.count_documents({"$text": {"$search": "noir"}}))          # all noir matches: 60
    # print(db.movies.count_documents({"$text": {"$search": "\"neo-noir\""}}))  # exact phrase neo-noir: 5
    # return list(db.movies.find({"$text": {"$search": "\"neo-noir\""}}, {"title": 1, "vote_count": 1}))
    return aggregate(db.movies, [
        # $text uses the text index on overview + tagline (MongoDB picks it: a
        # collection has only one text index). $search is a string of words,
        # OR'ed, case-insensitive. Only "noir": "neo-noir" would be split into
        # "neo" OR "noir" and also match e.g. Neo in The Matrix.
        # $text must be in the first stage; vote_count goes in the same $match.
        {"$match": {"$text": {"$search": "noir"}, "vote_count": {"$gte": 50}}},
        # Highest score first; ties: more votes (more reliable), then id
        {"$sort": {"vote_average": -1, "vote_count": -1, "_id": 1}},
        {"$limit": 20},
        {"$project": {"_id": 0, "title": 1, "year": {"$toString": {"$year": "$release_date"}},
                      "vote_average": 1, "vote_count": 1}},
    ])


def q8(db: Database) -> list[Row]:
    """Which 20 director-actor pairs with >= 3 collaborations (same movie) have
    the highest mean vote_average, considering only movies with vote_count >=
    100? Include the pair's films count and mean revenue.

    Decided: a director who acts in their own movie is not paired with
    themselves (3,754 movies have such a director). Directors: crew with
    job = "Director"; actors: cast.
    Columns: director, actor, films, mean_vote_average, mean_revenue.
    """
    # The cast trimmed to {id, name} per actor (see Q2)
    cast_ids_names = {"$map": {"input": "$cast", "as": "c",
                               "in": {"id": "$$c.id", "name": "$$c.name"}}}
    return aggregate(db.movies, [
        # Movies with >= 100 votes and at least one director.
        {"$match": {"vote_count": {"$gte": 100}, "crew.job": "Director"}},
        # One document per (movie, director), the same three steps as Q1
        {"$unwind": "$crew"},
        {"$match": {"crew.job": "Director"}},
        # Keep what the group needs; the cast becomes a small {id, name} list
        {"$project": {"_id": 1, "vote_average": 1, "vote_count": 1, "revenue": 1,
                      "actor": cast_ids_names, "crew": 1}},
        # One document per (movie, director, actor)
        {"$unwind": "$actor"},
        # A director who also acts in the movie is not paired with themselves.
        {"$match": {"$expr": {"$ne": ["$actor.id", "$crew.id"]}}},
        # One document per (director, actor) pair. The cast has one entry per
        # person per movie, so each film counts once per pair.
        # $avg skips null: mean_revenue is over the films with known revenue
        # (null if none of them has one).
        {"$group": {
            "_id": {"director": "$crew.id", "actor": "$actor.id"},
            "director": {"$first": "$crew.name"},
            "actor": {"$first": "$actor.name"},
            "films": {"$sum": 1},
            "mean_vote_average": {"$avg": "$vote_average"},
            "mean_revenue": {"$avg": "$revenue"},
        }},
        # HAVING films >= 3
        {"$match": {"films": {"$gte": 3}}},
        # Highest mean score first; ties: longer collaboration first, then ids
        {"$sort": {"mean_vote_average": -1, "films": -1, "_id": 1}},
        {"$limit": 20},
        {"$project": {"_id": 0, "director": 1, "actor": 1, "films": 1,
                      "mean_vote_average": 1, "mean_revenue": 1}},
    ])


def q9(db: Database) -> list[Row]:
    """Among movies where original_language != "en" but at least one production
    company or country is United States, which are the top 10 original
    languages by count? For each language, report the count and one example
    title.

    Decided: production companies carry no country, so "United States" is
    matched on production_countries (iso_3166_1 == "US"). "xx" (TMDB's code for
    no language) is kept and shown as "xx (no language)".
    Columns: original_language, movies, example_title.
    """
    return aggregate(db.movies, [
        {"$match": {"original_language": {"$nin": ["en", None]},
                    "production_countries.iso_3166_1": "US"}},
        # One document per language; $first picks the first matching movie as
        # the example (title = English title, readable in the report)
        {"$group": {
            "_id": "$original_language",
            "movies": {"$sum": 1},
            "example_title": {"$first": "$title"},
        }},
        # Most movies first; ties alphabetically by language code
        {"$sort": {"movies": -1, "_id": 1}},
        {"$limit": 10},
        # Output columns in this order. $cond is if/else: [if, then, else],
        # used to label TMDB's "xx" code
        {"$replaceWith": {
            "original_language": {"$cond": [{"$eq": ["$_id", "xx"]}, "xx (no language)", "$_id"]},
            "movies": "$movies",
            "example_title": "$example_title",
        }},
    ])


def user_stats() -> list[dict[str, Any]]:
    """Stages shared by Q10a and Q10b: one small document per user with >= 20 ratings.

    No $group is needed: the users collection already has one document per
    user, so everything is computed inside each document.
    """
    return [
        # >= 20 ratings: "ratings.19" is the 20th element (index 19), so it
        # exists only if the array has at least 20 elements. Cheaper than
        # counting the whole array.
        {"$match": {"ratings.19": {"$exists": True}}},
        # Replace each user with just the numbers. This drops the big ratings
        # array right away, so only small documents move on (and only 10 of
        # them reach Python). Inside the expressions, "$ratings" is still the
        # INPUT document's array.
        #   variance          $stdDevPop over the ratings, squared with $pow
        #                     (population variance, not $stdDevSamp)
        #   distinct_genres   "$ratings.genres.id" goes through two array levels
        #                     and gives one id list per rating, e.g.
        #                     [[16, 35], [12, 14], [35]]; $reduce + $setUnion
        #                     flattens it to distinct ids (as in Q3)
        {"$replaceWith": {
            "user": {"$toString": "$_id"},          # string: prints "45811", not "45,811"
            "ratings": {"$size": "$ratings"},
            "variance": {"$pow": [{"$stdDevPop": "$ratings.rating"}, 2]},
            "distinct_genres": {"$size": {"$reduce": {
                "input": "$ratings.genres.id",
                "initialValue": [],
                "in": {"$setUnion": ["$$value", "$$this"]},
            }}},
        }},
    ]


# Filled by the first Q10 call, reused by the second, so users is only read once.
_q10_results: dict[str, list[Row]] = {}


def q10_both(db: Database) -> dict[str, list[Row]]:
    """Compute both Q10 top-10 lists in ONE pass over users.

    $facet runs several sub-pipelines on the same input documents: user_stats()
    reads the 26M ratings once, then each facet only sorts the small per-user
    documents. The result is a single document {"genre_diverse": [...],
    "high_variance": [...]}, stored in _q10_results so the second call is free.
    """
    if not _q10_results:
        result = aggregate(db.users, user_stats() + [
            {"$facet": {
                # Most distinct genres first; ties (many users have all 20):
                # more ratings first. $sort directly followed by $limit only
                # keeps the current top 10 in memory.
                "genre_diverse": [
                    {"$sort": {"distinct_genres": -1, "ratings": -1, "user": 1}},
                    {"$limit": 10},
                ],
                # Highest variance first. Maximum possible on a 0.5-5.0 scale:
                # half 0.5 and half 5.0 -> 2.25^2 = 5.0625
                "high_variance": [
                    {"$sort": {"variance": -1, "user": 1}},
                    {"$limit": 10},
                ],
            }},
        ])
        # $facet returns ONE document holding both lists
        _q10_results.update(result[0])
    return _q10_results


def q10_genre_diverse(db: Database) -> list[Row]:
    """Q10 (part 1): top 10 most genre-diverse users (>= 20 ratings)."""
    return q10_both(db)["genre_diverse"]


def q10_high_variance(db: Database) -> list[Row]:
    """Q10 (part 2): top 10 highest-variance users (>= 20 ratings)."""
    return q10_both(db)["high_variance"]


# Order and titles used by task2.py
QUERIES: dict[str, tuple[str, Any]] = {
    "1": ("Top 10 directors (>= 5 movies) by median revenue", q1),
    "2": ("Actor pairs with >= 3 movies together", q2),
    "3": ("Top 10 actors (>= 10 movies) by genre breadth", q3),
    "4": ("Top 10 collections (>= 3 movies) by total revenue", q4),
    "5": ("Median runtime and count by decade and primary genre", q5),
    "6": ("Female share of the top-5 billed cast, by decade", q6),
    "7": ("Top 20 'noir' / 'neo-noir' movies (vote_count >= 50)", q7),
    "8": ("Top 20 director-actor pairs (>= 3 films, vote_count >= 100)", q8),
    "9": ("Top 10 non-English original languages with a US production country", q9),
    "10a": ("Top 10 most genre-diverse users (>= 20 ratings)", q10_genre_diverse),
    "10b": ("Top 10 highest-variance users (>= 20 ratings)", q10_high_variance),
}