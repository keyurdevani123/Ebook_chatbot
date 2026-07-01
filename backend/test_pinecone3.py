import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv("g:/ebook-chat/backend/.env")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("ebooks-production")

# Random non-zero vector
vec = [0.1]*384
vec[0] = 0.5

res1 = index.query(vector=[vec], top_k=2, include_metadata=True)
print("res1 (list of list) matches:")
for m in res1.matches:
    print(m.id, m.score)

res2 = index.query(vector=vec, top_k=2, include_metadata=True)
print("res2 (list of float) matches:")
for m in res2.matches:
    print(m.id, m.score)
