# TDT4225 Assignment 3 - MongoDB

Python code for analysing, cleaning and loading The Movies Dataset into MongoDB
(Part 1), and for answering the ten questions with aggregation pipelines (Part 2).

## Layout

```
.
├── docker-compose.yml     # MongoDB 7.0.1
├── docker/mongo-init.js   # creates the app user on first start
├── .env                   # connection settings (copy from .env.example, not committed)
├── requirements.txt
├── pyrightconfig.json     # type checking settings for Pylance / pyright
├── task1.py               # Part 1: EDA, cleaning and insertion
├── task2.py               # Part 2: runs the queries and prints the results
├── data/                  # dataset CSVs (not committed)
├── output/                # plots from the EDA (not committed)
└── src/
    ├── DbConnector.py     # MongoDB connection (reads .env)
    ├── paths.py           # project paths (data/, output/, .env)
    ├── load.py            # reading the CSVs (everything as text, ratings in chunks)
    ├── repair.py          # stitches the 3 records split by a line break
    ├── links.py           # rules for links.csv (shared by EDA and cleaning)
    ├── clean.py           # cleaning: CSVs -> movie and user documents
    ├── insert.py          # inserting, indexes and verification
    ├── schema.py          # document shapes as TypedDicts
    ├── queries.py         # the ten aggregation pipelines + explain helper
    ├── example.py         # the course's pymongo example
    └── eda/               # exploratory data analysis
        ├── run.py         # runs all EDA steps in order
        ├── core.py        # one section per CSV
        ├── common.py      # printing / plotting helpers
        └── broken_rows.py, small_values.py, links_check.py, lost_ratings.py,
            credits_dups.py, quality.py, ratings_sorted.py   # extra checks
```

## Data

Download The Movies Dataset from Kaggle
(https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and put the CSVs in
`data/`:

```
movies_metadata.csv  credits.csv  keywords.csv  links.csv  ratings.csv
links_small.csv  ratings_small.csv    # only needed for --small
```

## Setup

Run these commands from the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` if it does not already exist, and set the database
connection values. The Python code always reads `.env` from the project root,
regardless of which directory you run from.

```bash
docker compose up -d
docker compose ps        # wait until STATUS says (healthy)
```

The app user in `.env` is only created the first time the container starts with an
empty volume. If you change `DB_USER`/`DB_PASSWORD` later, reset with
`docker compose down -v` (this deletes the data).

## Running

Always run from the project root.

### Part 1: EDA, cleaning and insertion

```bash
python task1.py               # EDA, then clean + insert the full dataset
python task1.py --skip-eda    # only clean + insert
python task1.py --eda-only    # only the EDA
python task1.py --no-ratings  # EDA without the (slow) ratings passes
python task1.py --small       # use links_small.csv / ratings_small.csv while developing
```

The insert drops and recreates the `movies` and `users` collections, so it can be
rerun safely. The load ends with a verification
(counts, document sizes, types, text index) that should report
`14/14 checks passed`.

Collections:

- `movies`: one document per movie, `_id` = TMDB id, with genres, keywords, cast
  (one entry per person) and crew embedded. Text index on `overview` + `tagline`.
- `users`: one document per user, `_id` = MovieLens userId, with an embedded
  `ratings` array; each rating holds the TMDB `movie_id` and a copy of the movie's
  genres.

See `src/schema.py` for every field.

### Part 2: queries

```bash
python task2.py              # all questions
python task2.py 1 7          # only Q1 and Q7
python task2.py 10           # Q10 (both parts: 10a and 10b)
python task2.py 2 --rows 50  # print more rows of a long result (default 25)
python task2.py 7 --pprint   # also pprint the raw documents from MongoDB
python task2.py 7 --explain  # also print MongoDB's execution plan (index used, docs read, time)
```

Each question is one function in `src/queries.py`. Q2 and Q10a are the slow ones
(about 1.5 minutes each on the full dataset); Q10a and Q10b share one pass over the
users collection, so 10b is instant after 10a. `--explain` runs each query once more,
so it roughly doubles the time.

### Other

```bash
python -m src.example   # modules inside src/ are run with -m
```

Open a Mongo shell in the container (user and password from `.env`):

```bash
docker exec -it tdt4225-mongo mongosh -u tdt4225 -p tdt4225 --authenticationDatabase movie movie
```

## Type checking (optional)

The code has type hints for VS Code / Pylance. To check from the terminal:

```bash
python -m pip install pyright pandas-stubs
pyright
```