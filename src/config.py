import os
from dotenv import load_dotenv

# Load environment variables from the .env file into the system environment
load_dotenv()

# Retrieve the Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Fail fast if the key is missing, instead of failing later with a confusing error
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found! Please check that your .env file contains "
        "a line like: GROQ_API_KEY=your_key_here"
    )

# Name of the Groq-hosted LLM used throughout the project
LLM_MODEL_NAME = "openai/gpt-oss-20b"

# Directory where the Chroma vector store is persisted on disk
VECTOR_STORE_DIR = "data/vector_store"

# Chunk size (in characters) used when splitting documents for embedding
CHUNK_SIZE = 1000

# Overlap (in characters) between consecutive chunks, to preserve context
CHUNK_OVERLAP = 150