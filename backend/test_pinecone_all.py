import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv("g:/ebook-chat/backend/.env")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("ebooks-production")

res = index.query(vector=[0.0]*384, top_k=20, include_metadata=True)
for m in res.matches:
    print(m.id, {k: v for k, v in m.metadata.items() if k != "text"})
