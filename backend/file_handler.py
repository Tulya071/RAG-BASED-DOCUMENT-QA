from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def load_and_split(file_bytes: bytes, filename: str):
    ext = os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == ".txt":
            loader = TextLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks