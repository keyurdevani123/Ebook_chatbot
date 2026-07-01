from .base import Base


class CourseMessages(Base):

    def __init__(self) -> None:
        super().__init__()

        self.collection_name = "course_messages"

    def store_course_message(self, text, user_id: str, course_id: int, type: str, lang: str | None, model: str) -> dict:
        return self.insert_one(
            {
                "text": text,
                "user_id": user_id,
                "course_id": course_id,
                "type": type,
                "lang": lang,
                "model": model,
            }
        )

    def get_course_messages(self, user_id: str, course_id: int, skip: int = 0, limit: int = 10) -> list:
        docs = self.find_by_query({"user_id": user_id, "course_id": course_id}, limit=limit, skip=skip)

        for doc in docs:
            doc["_id"] = str(doc["_id"])

        return docs

    def get_last_six_course_messages(self, user_id: str, course_id: int, skip: int = 0) -> list:
        docs = self.find_by_query({"user_id": user_id, "course_id": course_id}, limit=6, skip=skip)

        messages = []
        for doc in docs:
            messages.append({"role": doc["type"].lower(), "content": doc["text"]})

        return messages

    def delete_course_messages(self, user_id: str, course_id: int) -> int:
        deleted_count = self.delete_by_query({"user_id": user_id, "course_id": course_id})

        return deleted_count
