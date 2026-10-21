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

Useful building blocks: $match, $project / $addFields, $unwind, $group,
$sort, $limit, $lookup, $facet, $size, $slice, $arrayElemAt, $year,
$filter, $setUnion / $addToSet, $median (MongoDB 7.0+), $stdDevPop.
Pass allowDiskUse=True to aggregate() for heavy pipelines (Q2, Q8).
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
        {"$match": {"crew.job": "Director"}},   #Finds movies that has atleast one director
        {"$unwind": "$crew"},                   #Unwind, toy story becomes 100 documents, one for each crew
        {"$match": {"crew.job": "Director"}},   #Finaly mach for director, now we will have one document for each director
        {"$group": {                            #We now group it based on director id, and accumliate and calculate all needed values
            "_id": "$crew.id",
            "director": {"$first": "$crew.name"},
            "movies": {"$sum": 1},
            "median_revenue": {"$median": {"input": "$revenue", "method": "approximate"}},
            "mean_vote_average": {"$avg": "$vote_average"},
        }},
        {"$match": {"movies": {"$gte": 5}}},            #Only look at directors that has more than 5 movies
        {"$sort": {"median_revenue": -1, "_id": 1}},    #Sort on median revenue falling
        {"$limit": 10},                                 #Limit to 10
        {"$project": {"_id": 0, "director": 1, "movies": 1,
                      "median_revenue": 1, "mean_vote_average": 1}}, # what is returned, turns of _id
    ])


def q2(db: Database) -> list[Row]:
    """List actor pairs that have co-starred in >= 3 movies together, their
    number of co-appearances, and their average movie vote_average. Sort by
    number of co-appearances.

    Notes: count each pair once - (A, B) and (B, A) are the same pair, so keep
    the person with the smaller id first. Big casts give many pairs (n*(n-1)/2
    per movie), so this is the heaviest query. The result can be long: return
    it all, task2.py prints the first rows and the total.
    Columns: actor_1, actor_2, movies, mean_vote_average.
    """

    cast_ids_names = {"$map": {"input": "$cast", "as": "c",
                               "in": {"id": "$$c.id", "name": "$$c.name"}}}
    # since we made one actor only have one entry for each movie it becomes easier to not pair actors with themself                       
    return aggregate(db.movies, [
        {"$project": {"vote_average": 1, "a": cast_ids_names, "b": cast_ids_names}},
        {"$unwind": "$a"},
        {"$unwind": "$b"}, 
        {"$match": {"$expr": {"$lt": ["$a.id", "$b.id"]}}},
        {"$group": {
            "_id": {"a": "$a.id", "b": "$b.id"},
            "actor_1": {"$first": "$a.name"},
            "actor_2": {"$first": "$b.name"},
            "movies": {"$sum": 1},
            "avg_vote_average": {"$avg": "$vote_average"},
        }},
        {"$match": {"movies": {"$gte": 3}}},
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
        # one document per (movie, actor)
        {"$unwind": "$cast"},
        # one document per actor: movie count + one genre-name list per movie
        {"$group": {
            "_id": "$cast.id",
            "actor": {"$first": "$cast.name"},
            "movies": {"$sum": 1},
            "genres": {"$addToSet": "$genres.name"},
        }},
        # >= 10 credited movies (before the $reduce, so it runs for fewer actors)
        {"$match": {"movies": {"$gte": 10}}},
        # flatten the lists of genre names into one list of distinct genres
        {"$addFields": {"genres": {"$reduce": {
            "input": "$genres",
            "initialValue": [],
            "in": {"$setUnion": ["$$value", "$$this"]},
        }}}},
        {"$addFields": {"distinct_genres": {"$size": "$genres"}}},
        # below is used to find how many acctors that had 20 geners, not used in the final query
        # {"$match": {"distinct_genres": 20}},
        # {"$count": "actors_with_all_genres"},
        # widest breadth first; ties: more movies first, then actor id
        {"$sort": {"distinct_genres": -1, "movies": -1, "_id": 1}},
        {"$limit": 10},
        {"$project": {"_id": 0, "actor": 1, "movies": 1, "distinct_genres": 1,
                      "example_genres": {"$slice": ["$genres", 5]}}},
    ])


def q4(db: Database) -> list[Row]:
    """For film collections (belongs_to_collection.name not null) with >= 3
    movies, which 10 collections have the largest total revenue? For each,
    report movie count, total revenue, median vote_average, and the earliest
    -> latest release date.

    Columns: collection, movies, total_revenue, median_vote_average,
    first_release, last_release.
    """
    return aggregate(db.movies, [
        # $match on belongs_to_collection.name.
        {"$match": {"belongs_to_collection.name": {"$ne": None}}},

        {"$group": {
            "_id": "$belongs_to_collection.id",
            "collection": {"$first": "$belongs_to_collection.name"},
            "movies": {"$sum": 1},
            "genres": {"$addToSet": "$genres.name"},
            "total_revenue": {"$sum": "$revenue"},
            "median_vote_average": {"$median": {"input": "$vote_average", "method": "approximate"}},
            "first_release": {"$min": "$release_date"},
            "last_release": {"$max": "$release_date"},
        }},

        {"$match": {"movies": {"$gte": 3}}},
        {"$sort": {"total_revenue": -1, "movies": -1, "_id": 1}},
        {"$limit": 10},
        {"$project": {"_id": 0, "collection": 1, "movies": 1, "total_revenue": 1,
                      "median_vote_average": 1, "release_span": {"$concat": [
                    {"$dateToString": {"date": "$first_release", "format": "%Y-%m-%d"}},
                    " -> ",
                    {"$dateToString": {"date": "$last_release", "format": "%Y-%m-%d"}},
                    ]},}},
    ])


# Q2, Q3, Q8: $unwind + $group, here on cast instead of crew.
# Q3: $addToSet collects distinct values.
# Q4: a $match on belongs_to_collection.name.
# Q5, Q6: $year on release_date, then subtract the remainder to get the decade.

def q5(db: Database) -> list[Row]:
    """By decade (1970s, 1980s, ...) and primary genre (first element in
    genres), what is the median runtime and movie count? Sort by decade, then
    median runtime (desc).

    Notes: movies without a release date or without genres cannot be placed and
    drop out; runtime null is ignored by $median. Decade = year - year % 10.
    Columns: decade, primary_genre, median_runtime, movies.
    """
    return aggregate(db.movies, [
        {"$match": {"release_date": {"$ne": None}, "genres.0": {"$exists": True}}},
        {"$addFields": {
            "decade": {"$subtract": [{"$year": "$release_date"},
                                     {"$mod": [{"$year": "$release_date"}, 10]}]},
            "primary_genre": {"$arrayElemAt": ["$genres.name", 0]},
        }},
        {"$group": {
            "_id": {"decade": "$decade", "primary_genre": "$primary_genre"},
            "median_runtime": {"$median": {"input": "$runtime", "method": "approximate"}},
            "movies": {"$sum": 1},
        }},
        {"$sort": {"_id.decade": 1, "median_runtime": -1, "_id.primary_genre": 1}},
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
        {"$addFields": {
            "decade": {"$subtract": [{"$year": "$release_date"},
                            {"$mod": [{"$year": "$release_date"}, 10]}]},
            # genders of the top 5, unknown (0) removed
            "genders": {"$filter": {"input": {"$slice": ["$cast.gender", 5]}, "as": "g",
                                    "cond": {"$in": ["$$g", [1, 2]]}}},
        }},
        # a movie with no known gender in its top 5 has no proportion -> drop it
        {"$match": {"genders.0": {"$exists": True}}},
        {"$addFields": {"female_proportion": {"$divide": [
            {"$size": {"$filter": {"input": "$genders", "as": "g", "cond": {"$eq": ["$$g", 1]}}}},
            {"$size": "$genders"},
        ]}}},

        {"$group": {
            "_id": "$decade",
            "avg_female_proportion": {"$avg": "$female_proportion"},
            "movies": {"$sum": 1},
        }},
        {"$sort": {"avg_female_proportion": -1, "_id": 1}},
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
    Columns: title, year, vote_average, vote_count.
    """
    # print(db.movies.count_documents({"$text": {"$search": "noir"}}))                 # all noir matches, any vote count
    # print(db.movies.count_documents({"$text": {"$search": "\"neo-noir\""}}))          # exact phrase neo-noir, any vote count
    # return list(db.movies.find({"$text": {"$search": "\"neo-noir\""}}, {"title": 1, "vote_count": 1}))
    return aggregate(db.movies, [
        {"$match": {"$text": {"$search": "noir"}, "vote_count": {"$gte": 50}}},
        {"$sort": {"vote_average": -1, "vote_count": -1, "_id": 1}},
        {"$limit": 20},
        {"$project": {"_id": 0, "title": 1, "year": {"$year": "$release_date"},
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
    cast_ids_names = {"$map": {"input": "$cast", "as": "c",
                               "in": {"id": "$$c.id", "name": "$$c.name"}}}
    # since we made one actor only have one entry for each movie it becomes easier to not pair actors with themself                       
    return aggregate(db.movies, [
        {"$match": {"vote_count": {"$gte": 100}, "crew.job": "Director"}},
        {"$unwind": "$crew"},                   #Unwind, toy story becomes 100 documents, one for each crew
        {"$match": {"crew.job": "Director"}},   #Finaly mach for director, now we will have one document for each director
        {"$project": {"_id": 1, "vote_average": 1, "vote_count": 1, "revenue": 1, "actor": cast_ids_names, "crew": 1}},
        {"$unwind": "$actor"},
        {"$match": {"$expr": {"$ne": ["$actor.id", "$crew.id"]}}},
        {"$group": {
            "_id": {"director": "$crew.id", "actor": "$actor.id"},
            "director": {"$first": "$crew.name"},
            "actor": {"$first": "$actor.name"},
            "films": {"$sum": 1},
            "mean_vote_average": {"$avg": "$vote_average"},
            "mean_revenue": {"$avg": "$revenue"},
        }},
        {"$match": {"films": {"$gte": 3}}},
        {"$sort": {"mean_vote_average": -1, "films": -1, "_id": 1}},
        {"$limit": 20},
        {"$project": {"_id": 0, "director": 1, "actor": 1, "films": 1, "mean_vote_average": 1, "mean_revenue": 1}},     
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
    raise NotImplementedError


def q10_genre_diverse(db: Database) -> list[Row]:
    """Q10 (part 1): for each user - ratings count, population variance of
    ratings, number of distinct genres rated. Top 10 most genre-diverse users
    (only users with >= 20 ratings).

    Notes: users collection; distinct genres by genre id. Population variance
    = $stdDevPop squared (not $stdDevSamp).
    Columns: user, ratings, variance, distinct_genres.
    """
    raise NotImplementedError


def q10_high_variance(db: Database) -> list[Row]:
    """Q10 (part 2): the same per-user numbers, top 10 highest-variance users
    (only users with >= 20 ratings).

    Columns: user, ratings, variance, distinct_genres.
    """
    raise NotImplementedError


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