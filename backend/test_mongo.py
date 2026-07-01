import os
from dotenv import load_dotenv
from ebook_backend.database.files import UserFiles
from ebook_backend.database.sessions import ChatSessions

load_dotenv("g:/ebook-chat/backend/.env")

files_db = UserFiles()
docs = files_db.find_by_query({})
print("All files in DB:", docs)

sessions_db = ChatSessions()
sdocs = sessions_db.find_by_query({})
print("All sessions in DB:", sdocs)
