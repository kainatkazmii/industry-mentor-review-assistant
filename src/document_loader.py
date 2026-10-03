import os
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader


def load_docx(file_path: str) -> str:
    """
    Takes the path of a .docx file and extracts its full plain text.
    """
    loader = Docx2txtLoader(file_path)
    documents = loader.load()

    # Docx2txtLoader normally returns a single Document (the whole file at once)
    # We join all documents' text, to be future-proof
    full_text = "\n".join(doc.page_content for doc in documents)
    return full_text


def load_pdf(file_path: str) -> str:
    """
    Takes the path of a .pdf file and extracts its full plain text.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # PyPDFLoader returns a separate Document for each PAGE
    # So we join the text of all pages, with a page-break separator
    full_text = "\n\n".join(doc.page_content for doc in documents)
    return full_text


def load_document(file_path: str) -> str:
    """
    Checks the file extension and calls the correct loader function.
    This is the "entry point" - other files should only call this.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()  # so ".DOCX" and ".docx" both work

    if extension == ".docx":
        return load_docx(file_path)
    elif extension == ".pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Only .docx and .pdf files are supported."
        )