from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_with_langchain(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def create_documents_from_text(text: str, chunk_size: int = 700, title: str = "transcript") -> List[Document]:
    chunks = split_with_langchain(text, chunk_size)
    return [
        Document(page_content=chunk, metadata={"title": title, "chunk_index": idx})
        for idx, chunk in enumerate(chunks)
    ]