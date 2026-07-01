import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv("g:/ebook-chat/backend/.env")

client = MongoClient(os.environ.get("MONGODB_HOST", "mongodb://localhost:27017"))
print("Databases:", client.list_database_names())
for db_name in client.list_database_names():
    db = client[db_name]
    print(f"DB {db_name} Collections:", db.list_collection_names())
    for coll in db.list_collection_names():
        print("  ", coll, "count:", db[coll].count_documents({}))
