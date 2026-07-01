from typing import List

from langchain.schema.document import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..models.document import VectorDocument


class OpenAIClient:

    def __init__(self) -> None:
        self.model = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed_documents(self, docs: List[Document]) -> List:
        texts = [doc.page_content for doc in docs]

        vectors = self.embeddings.embed_documents(texts)

        return [
            VectorDocument(vector=vector, metadata={**doc.metadata, "text": doc.page_content})
            for vector, doc in zip(vectors, docs)
        ]

    def embed_query(self, query: str) -> List:
        return self.embeddings.embed_query(query)

    def chat(self, input: LanguageModelInput, model: str) -> str:

        chat_model = ChatOpenAI(model=model, temperature=0.6)

        if isinstance(input, tuple) and len(input) == 2:
            input = [SystemMessage(input[0]), HumanMessage(input[1])]

        return chat_model.invoke(input).content
