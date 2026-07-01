from ebook_backend.database.files import UserFiles
from dotenv import load_dotenv
load_dotenv("g:/ebook-chat/backend/.env")

files_db = UserFiles()
doc_id = files_db.store_file("test_user", "test.pdf", "test/path", 100)
print("Stored doc_id:", doc_id)
docs = files_db.get_user_files("test_user")
print("Retrieved docs:", docs)
