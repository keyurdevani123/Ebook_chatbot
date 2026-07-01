import datetime

from dotenv import load_dotenv

from ebook_backend.doc_handler import split_text
from ebook_backend.openai import OpenAIClient
from ebook_backend.store import PineconeDBClient
from training.courses.transcript_handler import get_transcript_content_and_metadata

# Load environment variables from .env file
load_dotenv()

# Global variables
COURSES = {
    # 183173: 19981430,  # security+7
    158276: 19981640,  # CompTIA A+ (220-1101/2)
    186273: 20693228,  # CompTIA Network+ (N10-009)
    168549: 19981440,  # CompTIA Project+ (PK0-005)
    180210: 19981449,  # CompTIA CySA+ (CS0-003)
    149336: 19981785,  # CompTIA Server+ (SK0-005)
    160465: 19981547,  # CompTIA Linux+ (XK0-005)
    164781: 19981424,  # (ISC)² Systems Security Certified Practitioner (SSCP)
    177936: 19981750,  # ISACA Certified Information Security Manager (CISM)
    144663: 19981771,  # NIST Cybersecurity and Risk Management Frameworks
    181630: 19981439,  # ISACA Certified in Risk and Information Systems Control (CRISC)
    174967: 19981461,  # Certified Associate in Project Management (CAPM) 2023
    182890: 19981434,  # Certified Ethical Hacker (CEH) v.12
    182845: 19981438,  # MS-102: Microsoft 365 Administrator
    174966: 19981453,  # Microsoft Azure Administrator (AZ-104)
    183413: 19981433,  # MD-102: Endpoint Administrator
    178061: 19981452,  # Designing Microsoft Azure Infrastructure Solutions (AZ-305)
    178060: 19981451,  # Developing Solutions for Microsoft Azure (AZ-204)
    180684: 19981447,  # SC-200: Microsoft Security Operations Analyst
}


print(f"{datetime.datetime.now().isoformat()} | Start of the training script")
print(f"{datetime.datetime.now().isoformat()} | List of courses to be trained: {COURSES}")

# Initialize client instances
openai_client = OpenAIClient()
vectordb_client = PineconeDBClient(collection_name="production-video-courses")
vectordb_client.initialize_collection()


def train_single(course_id, project_id):

    print(f"{datetime.datetime.now().isoformat()} | Training | COURSE_ID: {course_id} | Training video course...")

    for transcript in get_transcript_content_and_metadata(course_id, project_id):

        text = transcript["text"]
        metadata = transcript["metadata"]
        video_id = metadata["video_id"]

        docs = split_text(text, metadata)

        print(
            f"{datetime.datetime.now().isoformat()} | Training | COURSE_ID: {course_id}"
            f" | VIDEO_ID: {video_id} | Chunks: {len(docs)}"
        )

        # # Debug logs
        for doc in [docs[-1]]:
            print(
                f"{datetime.datetime.now().isoformat()} | Training | COURSE_ID: {course_id}"
                f" | VIDEO_ID: {video_id} | Last document: {doc}"
            )

        # Generate embeddings from document chunks
        vectors_list = openai_client.embed_documents(docs)

        # Store points to vector store
        vectordb_client.save(vectors_list)

        print(
            f"{datetime.datetime.now().isoformat()} | Training | COURSE_ID: {course_id}"
            f" | VIDEO_ID: {video_id} | Video trained!"
        )
        print("=" * 60)


def train_all(courses):
    for course_id, project_id in courses.items():
        try:
            train_single(course_id, project_id)
        except Exception as err:
            print(
                f"{datetime.datetime.now().isoformat()} | Training | COURSE_ID: {course_id}"
                f" | Error training course: {err}"
            )
            # print(traceback.format_exc())


# Train all the courses by id and vimeo project id
train_all(COURSES)

print(f"{datetime.datetime.now().isoformat()} | End of the training script.")

# # Similarity search on doc to test
# QUERY = "Who is the author?"

# vector = openai_client.embed_query(QUERY)
# query_result = vectordb_client.query(vector=vector, course_id="SAMPLE1", limit=4)
# pprint(query_result)
