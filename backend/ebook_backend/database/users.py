import bcrypt
from .base import Base

class Users(Base):
    def __init__(self) -> None:
        super().__init__()
        self.collection_name = "users"

    def initialize_indexes(self) -> None:
        import pymongo
        collection = self.db[self.collection_name]
        collection.create_index([("email", pymongo.ASCENDING)], unique=True, background=True)

    def create_user(self, email: str, password: str, name: str) -> dict | None:
        # Check if user exists
        existing = self.find_by_query({"email": email})
        if existing:
            return None
            
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        user_doc = {
            "email": email,
            "password": hashed_pw,
            "name": name
        }
        user_id = self.insert_one(user_doc)
        return {"_id": str(user_id), "email": email, "name": name}

    def verify_user(self, email: str, password: str) -> dict | None:
        users = self.find_by_query({"email": email}, limit=1)
        if not users:
            return None
        user = users[0]
        if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return {"_id": str(user["_id"]), "email": user["email"], "name": user["name"]}
        return None
