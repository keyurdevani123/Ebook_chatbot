import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv("g:/ebook-chat/backend/.env")

client = MongoClient(os.environ.get("MONGODB_HOST", "mongodb://localhost:27017"))
db = client["ebook_db"]
print("Collections:", db.list_collection_names())
for coll in db.list_collection_names():
    print(coll, "count:", db[coll].count_documents({}))
