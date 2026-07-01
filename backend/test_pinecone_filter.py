import os
from dotenv import load_dotenv
from pinecone import Pinecone
from ebook_backend.database.files import UserFiles

load_dotenv("g:/ebook-chat/backend/.env")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("ebooks-production")

files_db = UserFiles()
docs = files_db.find_by_query({})
book_ids = [str(d["_id"]) for d in docs]
print("Book IDs in DB:", book_ids)

if book_ids:
    pinecone_filter = {"bookid": {"$in": book_ids}}
    res = index.query(vector=[0.0]*384, top_k=5, filter=pinecone_filter, include_metadata=True)
    print("Filter matches:", len(res.matches))
    if res.matches:
        print("First match metadata:", res.matches[0].metadata)
else:
    print("No book IDs to filter")
