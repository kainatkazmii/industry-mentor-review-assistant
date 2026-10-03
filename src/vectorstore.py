import os
import glob

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.document_loader import load_document


# Embedding model - converts text into numeric vectors
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Text splitter - breaks long documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def _load_all_files_from_folder(folder_path: str) -> list[str]:
    """
    Finds all .docx and .pdf files inside a folder and extracts the text
    of each one. Returns a list of strings (one per file).
    """
    file_paths = glob.glob(os.path.join(folder_path, "*.docx")) + \
                 glob.glob(os.path.join(folder_path, "*.pdf"))

    texts = []
    for path in file_paths:
        text = load_document(path)
        texts.append(text)
    return texts


def build_vectorstore(standards_folder: str, examples_folder: str) -> Chroma:
    """
    Reads all files from the standards and examples folders, splits them
    into chunks, computes embeddings, and saves everything into a Chroma
    vector database on disk.
    """
    standards_texts = _load_all_files_from_folder(standards_folder)
    examples_texts = _load_all_files_from_folder(examples_folder)

    all_texts = standards_texts + examples_texts

    all_chunks = []
    for text in all_texts:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)

    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_STORE_DIR,
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Loads an already-built Chroma database from disk (no need to
    rebuild chunks/embeddings if it was built before).
    """
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embedding_model,
    )
    return vectorstore


def get_relevant_chunks(query: str, vectorstore: Chroma, k: int = 4) -> list[str]:
    """
    Given a query, finds the top 'k' most relevant chunks in the vectorstore.
    """
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]