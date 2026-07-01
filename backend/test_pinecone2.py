import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv("g:/ebook-chat/backend/.env")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("ebooks-production")

vec = [0.1]*384

try:
    res1 = index.query(vector=[vec], top_k=2)
    print("res1 matches:", len(res1.matches))
except Exception as e:
    print("res1 error", e)

try:
    res2 = index.query(vector=vec, top_k=2)
    print("res2 matches:", len(res2.matches))
except Exception as e:
    print("res2 error", e)
