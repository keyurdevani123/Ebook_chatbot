from .base import Base


class Messages(Base):

    def __init__(self) -> None:
        super().__init__()

        self.collection_name = "messages"

    def initialize_indexes(self) -> None:
        import pymongo
        collection = self.db[self.collection_name]
        collection.create_index(
            [("user_id", pymongo.ASCENDING), ("book_id", pymongo.ASCENDING), ("createdAt", pymongo.DESCENDING)],
            background=True
        )

    def store_message(self, text, user_id: str, book_id: str, type: str, lang: str | None, model: str) -> dict:
        return self.insert_one(
            {
                "text": text,
                "user_id": user_id,
                "book_id": book_id,
                "type": type,
                "lang": lang,
                "model": model,
            }
        )

    def get_messages(self, user_id: str, book_id: str, skip: int = 0, limit: int = 10) -> list:
        docs = self.find_by_query({"user_id": user_id, "book_id": book_id}, limit=limit, skip=skip)

        for doc in docs:
            doc["_id"] = str(doc["_id"])

        return docs

    def get_last_six_messages(self, user_id: str, book_id: str, skip: int = 0) -> list:
        docs = self.find_by_query({"user_id": user_id, "book_id": book_id}, limit=6, skip=skip)

        messages = []
        for doc in docs:
            role = "user" if doc["type"] == "Human" else "assistant"
            messages.append({"role": role, "content": doc["text"]})

        return messages

    def delete_messages(self, user_id: str, book_id: str) -> int:
        deleted_count = self.delete_by_query({"user_id": user_id, "book_id": book_id})

        return deleted_count
