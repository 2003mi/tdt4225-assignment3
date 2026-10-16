import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from .paths import ENV_FILE

load_dotenv(ENV_FILE)


class DbConnector:
    """
    Connects to MongoDB. Defaults come from the .env file in the project root,
    but any value can be passed in explicitly.
    """

    client: MongoClient
    db: Database

    def __init__(self,
                 HOST: str | None = None,
                 DATABASE: str | None = None,
                 USER: str | None = None,
                 PASSWORD: str | None = None,
                 PORT: int | str | None = None) -> None:
        host: str = HOST or os.getenv("DB_HOST") or "127.0.0.1"
        database: str = DATABASE or os.getenv("DB_NAME") or "movie"
        user: str = USER or os.getenv("DB_USER") or "tdt4225"
        password: str | None = PASSWORD or os.getenv("DB_PASSWORD")
        port: int = int(PORT or os.getenv("DB_PORT") or 27017)

        if not password:
            raise RuntimeError(f"DB_PASSWORD is not set — check {ENV_FILE}")

        # The user is created in the database itself, so it is also the auth database
        uri = "mongodb://%s:%s@%s:%d/%s" % (
            quote_plus(user), quote_plus(password), host, port, database)

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[database]

        # MongoClient connects lazily; ping so bad credentials/host fail here
        try:
            self.client.admin.command("ping")
        except Exception as e:
            self.client.close()
            raise RuntimeError(f"Failed to connect to MongoDB at {host}:{port}: {e}") from e

        print("You are connected to the database:", self.db.name)
        print("-----------------------------------------------\n")

    def close_connection(self) -> None:
        self.client.close()
        print("\n-----------------------------------------------")
        print("Connection to %s-db is closed" % self.db.name)
