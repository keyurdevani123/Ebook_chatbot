from .base import Base

class UserFiles(Base):
    def __init__(self) -> None:
        super().__init__()
        self.collection_name = "user_files"

    def initialize_indexes(self) -> None:
        import pymongo
        collection = self.db[self.collection_name]
        collection.create_index(
            [("user_id", pymongo.ASCENDING), ("createdAt", pymongo.DESCENDING)],
            background=True
        )
        
        chunks_col = self.db["file_chunks"]
        chunks_col.create_index(
            [("text", pymongo.TEXT)],
            background=True
        )
        chunks_col.create_index(
            [("user_id", pymongo.ASCENDING), ("book_id", pymongo.ASCENDING)],
            background=True
        )

    def store_file(self, user_id: str, filename: str, filepath: str, size: int) -> dict:
        return self.insert_one(
            {
                "user_id": user_id,
                "filename": filename,
                "filepath": filepath,
                "size": size,
            }
        )

    def get_user_files(self, user_id: str) -> list:
        docs = self.find_by_query({"user_id": user_id}, limit=100)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        return docs

    def get_file(self, user_id: str, file_id: str) -> dict | None:
        from bson.objectid import ObjectId
        docs = self.find_by_query({"_id": ObjectId(file_id), "user_id": user_id})
        return docs[0] if docs else None
        
    def store_chunks(self, chunks: list[dict]) -> None:
        if not chunks: return
        # add createdAt manually if needed, or let DB handle
        import time
        now = int(time.time() * 1000)
        for c in chunks:
            if "createdAt" not in c:
                c["createdAt"] = now
        self.db["file_chunks"].insert_many(chunks)
        
    def keyword_search(self, book_ids: list[str], query: str, limit: int = 10) -> list[dict]:
        # MongoDB text search utilizing the $text index
        match_stage = {
            "book_id": {"$in": book_ids},
            "$text": {"$search": query}
        }
        pipeline = [
            {
                "$match": match_stage
            },
            {
                "$addFields": {
                    "score": {"$meta": "textScore"}
                }
            },
            {
                "$sort": {"score": -1}
            },
            {
                "$limit": limit
            }
        ]
        return list(self.db["file_chunks"].aggregate(pipeline))

    def delete_file(self, user_id: str, file_id: str) -> int:
        from bson.objectid import ObjectId
        return self.delete_by_query({"user_id": user_id, "_id": ObjectId(file_id)})
