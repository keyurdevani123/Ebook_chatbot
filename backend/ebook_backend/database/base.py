import os
import time

from pymongo import MongoClient


class Base:

    def __init__(self, connection_string: str = None) -> None:
        self.connection_string = connection_string or os.getenv("MONGODB_HOST", "mongodb://localhost:27017")
        self.db_name = "ebook_db"

        self.client = MongoClient(self.connection_string)
        self.db = self.client[self.db_name]

    def initialize_indexes(self) -> None:
        """Create indexes to optimize queries."""
        pass

    def insert_one(self, document: dict) -> dict:
        collection = self.db[self.collection_name]

        document = self.prepare_document(document)

        return collection.insert_one(document).inserted_id

    def find_by_query(self, query: dict, limit: int = 10, skip: int = 0) -> list:
        collection = self.db[self.collection_name]

        return list(collection.find(query, limit=limit, skip=skip, sort=[("createdAt", -1)]))

    def prepare_document(self, document: dict) -> dict:
        common = {"createdAt": int(time.time() * 1000)}

        return {
            **document,
            **common,
        }

    def delete_by_query(self, query: dict) -> int:
        collection = self.db[self.collection_name]

        return collection.delete_many(query).deleted_count
