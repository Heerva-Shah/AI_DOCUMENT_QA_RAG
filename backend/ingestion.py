# backend/ingestion.py

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import embeddings
import io

CHROMA_DIR = "./chroma_db"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def split_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def store_chunks(chunks: list[str], session_id: str) -> int:
    """Embed chunks and store them in ChromaDB under a session collection."""
    # Create collection
    vectorstore = Chroma(
        collection_name=f"session_{session_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    # Delete previous document for this session if any
    vectorstore.delete_collection()

    # Recreate fresh and store new chunks
    vectorstore = Chroma(
        collection_name=f"session_{session_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.add_texts(chunks)
    return len(chunks)


def ingest_document(file_bytes: bytes, session_id: str) -> dict:
    """Full ingestion pipeline: extract → split → embed → store."""
    text = extract_text_from_pdf(file_bytes)

    if not text:
        raise ValueError("Could not extract any text from this PDF. It may be a scanned image PDF.")

    chunks = split_text(text)
    num_chunks = store_chunks(chunks, session_id)

    return {
        "session_id": session_id,
        "num_chunks": num_chunks,
        "text_length": len(text)
    }