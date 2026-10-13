import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient

from .paths import ENV_FILE

load_dotenv(ENV_FILE)


class DbConnector:
    """
    Connects to MongoDB. Defaults come from the .env file in the project root,
    but any value can be passed in explicitly.
    """

    def __init__(self,
                 HOST=None,
                 DATABASE=None,
                 USER=None,
                 PASSWORD=None,
                 PORT=None):
        HOST = HOST or os.getenv("DB_HOST", "127.0.0.1")
        DATABASE = DATABASE or os.getenv("DB_NAME", "movie")
        USER = USER or os.getenv("DB_USER", "tdt4225")
        PASSWORD = PASSWORD or os.getenv("DB_PASSWORD")
        PORT = int(PORT or os.getenv("DB_PORT", 27017))

        if not PASSWORD:
            raise RuntimeError(f"DB_PASSWORD is not set — check {ENV_FILE}")

        # The user is created in DATABASE, so DATABASE is also the auth database
        uri = "mongodb://%s:%s@%s:%d/%s" % (
            quote_plus(USER), quote_plus(PASSWORD), HOST, PORT, DATABASE)

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[DATABASE]

        # MongoClient connects lazily; ping so bad credentials/host fail here
        try:
            self.client.admin.command("ping")
        except Exception as e:
            self.client.close()
            raise RuntimeError(f"Failed to connect to MongoDB at {HOST}:{PORT}: {e}") from e

        print("You are connected to the database:", self.db.name)
        print("-----------------------------------------------\n")

    def close_connection(self):
        self.client.close()
        print("\n-----------------------------------------------")
        print("Connection to %s-db is closed" % self.db.name)