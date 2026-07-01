from ebook_backend.database.files import UserFiles
from dotenv import load_dotenv
load_dotenv("g:/ebook-chat/backend/.env")

files_db = UserFiles()
docs = files_db.find_by_query({})
print("Files in DB:", docs)
