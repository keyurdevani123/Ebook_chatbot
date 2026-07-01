from .base import Base
from datetime import datetime
from bson.objectid import ObjectId


class ChatSessions(Base):
    def __init__(self) -> None:
        super().__init__()
        self.collection_name = "sessions"

    def initialize_indexes(self) -> None:
        import pymongo
        collection = self.db[self.collection_name]
        collection.create_index(
            [("user_id", pymongo.ASCENDING), ("createdAt", pymongo.DESCENDING)],
            background=True
        )

    def create_session(self, user_id: str, title: str) -> str:
        doc_id = self.insert_one(
            {
                "user_id": user_id,
                "title": title,
                "created_at": datetime.utcnow(),
            }
        )
        return str(doc_id)

    def get_user_sessions(self, user_id: str) -> list:
        docs = self.find_by_query({"user_id": user_id})
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            if "created_at" in doc:
                doc["created_at"] = doc["created_at"].isoformat()
        return docs

    def delete_session(self, user_id: str, session_id: str) -> int:
        return self.delete_by_query({"_id": ObjectId(session_id), "user_id": user_id})
    def update_one(self, query: dict, update: dict) -> bool:
        result = self.db[self.collection_name].update_one(query, update)
        return result.modified_count > 0
