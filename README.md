# TDT4225G04

Python code for loading and analysing the Movie dataset with MongoDB.

## Layout

```
.
├── docker-compose.yml   # MongoDB 7.0.1
├── docker/mongo-init.js # creates the app user on first start
├── .env                 # connection settings (copy from .env.example)
├── task1.py             # Part 1: EDA + insert
├── task2.py             # Part 2: queries
├── data/                # dataset CSVs (not committed)
└── src/                 # shared code (DbConnector, helpers, ...)
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

Always run from the project root:

```bash
python task1.py
python task2.py
python -m src.example   # modules inside src/ are run with -m
```

Open a Mongo shell in the container:

```bash
docker exec -it tdt4225-mongo mongosh -u tdt4225 -p tdt4225 --authenticationDatabase movie movie
```