"""Shapes of the data, as TypedDicts.

A TypedDict is a normal dict at runtime; the class only tells the editor (and
type checkers) which keys exist and what type each value has. Hover over a
variable typed as MovieDoc in VS Code and it lists every field; typing
movie["  suggests the keys.

Raw*   - objects as they appear inside the CSV files (after ast.literal_eval)
others - objects as they are stored in MongoDB after cleaning
"""
from datetime import datetime
from typing import TypedDict


# --- Raw objects from the CSV files ------------------------------------------

class RawCastEntry(TypedDict):
    """One element of credits.csv "cast"."""
    cast_id: int
    character: str
    credit_id: str
    gender: int          # 0 = unknown, 1 = female, 2 = male
    id: int              # TMDB person id
    name: str
    order: int           # billing order, 0 = top billed
    profile_path: str | None


class RawCrewEntry(TypedDict):
    """One element of credits.csv "crew"."""
    credit_id: str
    department: str
    gender: int
    id: int
    job: str             # e.g. "Director", "Producer"
    name: str
    profile_path: str | None


class RawCollection(TypedDict):
    """movies_metadata.csv "belongs_to_collection"."""
    id: int
    name: str
    poster_path: str | None
    backdrop_path: str | None


# --- Objects stored in MongoDB -----------------------------------------------

class Genre(TypedDict):
    id: int
    name: str


class Company(TypedDict):
    id: int
    name: str


class Country(TypedDict):
    iso_3166_1: str      # e.g. "US"
    name: str


class Language(TypedDict):
    iso_639_1: str       # e.g. "en"
    name: str


class Collection(TypedDict):
    id: int
    name: str


class Keyword(TypedDict):
    id: int
    name: str


class CastMember(TypedDict):
    """One person in a movie's cast (several roles are grouped)."""
    id: int
    name: str
    gender: int          # 0 = unknown, 1 = female, 2 = male
    order: int           # the person's best (lowest) billing order
    characters: list[str]


class CrewMember(TypedDict):
    id: int
    name: str
    job: str
    department: str


class MovieDoc(TypedDict):
    """A document in the movies collection."""
    _id: int                                  # TMDB id
    imdb_id: str | None
    title: str
    original_title: str
    original_language: str | None             # "xx" = no language
    overview: str | None
    tagline: str | None
    status: str | None
    release_date: datetime | None
    runtime: int | None                       # minutes
    budget: int | None                        # USD, null if unknown or < 1000
    revenue: int | None                       # USD, null if unknown or < 1000
    vote_average: float | None                # 0-10, null if no votes
    vote_count: int
    popularity: float | None
    adult: bool
    video: bool
    genres: list[Genre]                       # [0] is the primary genre
    belongs_to_collection: Collection | None
    production_companies: list[Company]
    production_countries: list[Country]
    spoken_languages: list[Language]
    keywords: list[Keyword]
    cast: list[CastMember]                    # sorted by order
    crew: list[CrewMember]


class Rating(TypedDict):
    """One rating inside a user document."""
    movie_id: int                             # TMDB id (= MovieDoc._id)
    rating: float                             # 0.5-5.0 in half steps
    timestamp: datetime                       # UTC
    genres: list[Genre]                       # copied from the movie


class UserDoc(TypedDict):
    """A document in the users collection."""
    _id: int                                  # MovieLens userId
    ratings: list[Rating]
