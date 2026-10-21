from pprint import pprint
from src.DbConnector import DbConnector

db = DbConnector().db
# duplicates = list(db.movies.aggregate([
#     {"$unwind": "$crew"},
#     {"$match": {"crew.job": "Director"}},
#     # one group per (movie, person): n = how many times that person is listed as Director
#     {"$group": {
#         "_id": {"movie": "$_id", "person": "$crew.id"},
#         "title": {"$first": "$title"},
#         "director": {"$first": "$crew.name"},
#         "n": {"$sum": 1},
#     }},
#     {"$match": {"n": {"$gt": 1}}},
# ]))
# print(f"{len(duplicates)} (movie, director) pairs listed more than once")
# pprint(duplicates[:10], sort_dicts=False)

pprint(db.movies.find_one({"title": "Senseless"}, {"title": 1, "original_language": 1, "spoken_languages": 1}))