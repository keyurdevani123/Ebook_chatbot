from ebook_backend.database.files import UserFiles
from dotenv import load_dotenv
load_dotenv("g:/ebook-chat/backend/.env")

files_db = UserFiles()
docs = files_db.find_by_query({"user_id": "6a44e1e175f0cdd3153cd0bb"})
print("Real User Files in DB:", docs)
