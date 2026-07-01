import os
import tempfile
from typing import List

import ebooklib
import openai
import pinecone
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ebooklib import epub
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.text import TextLoader
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-3.5-turbo", streaming=True)
DB_DIR = "ebook"

# OpenAI API setup
openai.api_key = os.getenv("OPENAI_API_KEY")


async def extract_text(epub_file: UploadFile) -> str:
    """
    Extracts text content from an EPUB file.

    Args:
        epub_file (UploadFile): The EPUB file to extract text from.

    Returns:
        str: The extracted text content.

    Raises:
        Exception: If an error occurs during the extraction process.
    """
    try:
        # Read the file content into a bytes object
        file_content = await epub_file.read()

        # Create a temporary file to write the bytes to
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
            tmp.write(file_content)
            # It's important to flush so all content is physically written to the disk
            tmp.flush()

            # Now you can use the temporary file's name to open it with epub.read_epub
            book = epub.read_epub(tmp.name)

        # Extract text from the epub content
        text = ""
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.content, "html.parser")
            text += soup.get_text() + " "

        # Optionally, write to another temporary file or handle the text as needed
        with open("temp.txt", "w", encoding="utf-8") as f:
            f.write(text)

        return text
    except Exception as e:
        print(e, "===")


@app.post("/create-embedding-ebook")
async def create_embedding_ebook(file: UploadFile = File(...), bookid: str = Form(...), bookname: str = Form(None)):
    """
    Endpoint to create embeddings for an ebook from an uploaded file.

    Args:
        file (UploadFile): The ebook file to process.
        bookid (str): The ID of the ebook.
        bookname (str, optional): The name of the ebook. Defaults to None.

    Returns:
        JSONResponse: JSON response indicating the status of the operation.

    Raises:
        Exception: If an error occurs during the process.
    """
    try:
        # Extract text from the uploaded ebook file
        await extract_text(file)

        # Load extracted text from the temporary file
        loader = TextLoader("temp.txt", encoding="utf-8")
        temp_data = loader.load()

        # Prepare metadata for the documents
        metadata = {"bookid": bookid, "bookname": bookname}

        # Create Document objects with metadata
        data = [Document(page_content=doc.page_content, metadata=metadata) for doc in temp_data]

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        # Initialize OpenAI embeddings with API key
        openai_embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

        # Compute embeddings for each document chunk
        vectordb = Chroma.from_documents(documents=docs, embedding=openai_embeddings, persist_directory=DB_DIR)

        # Persist the computed embeddings
        vectordb.persist()

        # Return success response
        return JSONResponse({"status": "Success"}, 201)
    except Exception as e:
        # Print and return failure response
        print(e)
        return JSONResponse({"status": "Failed"}, 500)


class ChatEbook(BaseModel):
    bookid: str
    prompt: str
    bookname: str


@app.post("/chat")
async def chat(request: ChatEbook):
    """
    Endpoint to perform a chat-based interaction.

    Args:
        request (Request): The request object containing the chat data.

    Returns:
        JSONResponse: JSON response containing the chat response.

    Raises:
        HTTPException: If an error occurs during the process.
    """
    try:
        # Parse the request body into a ChatEbook object (assuming ChatEbook is a Pydantic model)
        chat_data = request.dict()

        # Initialize the vector database with OpenAI embeddings
        db = Chroma(
            embedding_function=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]), persist_directory=DB_DIR
        )

        # Create a retriever object based on the user's bookid and bookname
        retriever = db.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {
                    "$and": [{"bookid": {"$eq": chat_data["bookid"]}}, {"bookname": {"$eq": chat_data["bookname"]}}]
                },
            }
        )

        # Retrieve relevant documents based on the user's prompt
        relevant_documents = retriever.get_relevant_documents(chat_data["prompt"])

        # Initialize a RetrievalQA object
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Generate a response based on the user's prompt
        response = qa(chat_data["prompt"])

        # Return the response
        return JSONResponse({"response": response}, 200)
    except Exception as e:
        # Print and raise an exception if an error occurs
        print(e)
        return JSONResponse({"error": "Something went wrong."}, 500)
