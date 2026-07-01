import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv("g:/ebook-chat/backend/.env")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("ebooks-production")
stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")

# query a dummy vector
res = index.query(vector=[0.0]*384, top_k=2, include_metadata=True)
print("Dummy query matches:")
for m in res.matches:
    print(m.id, m.score, {k: v for k,v in m.metadata.items() if k != "text"})
