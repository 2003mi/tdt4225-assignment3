"""Part 2: Queries."""
from src.DbConnector import DbConnector


def main():
    connection = DbConnector()
    try:
        db = connection.db
        # TODO: queries 1-10
    finally:
        connection.close_connection()


if __name__ == "__main__":
    main()