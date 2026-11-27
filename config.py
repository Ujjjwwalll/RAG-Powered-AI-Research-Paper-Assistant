"""
config.py - Configuration File
================================
What this file does:
    Holds ALL settings and constants for the entire project in one place.
    This way, if you need to change a setting, you change it here only.

Input:  Nothing (this file is imported by other modules)
Output: Constants and settings that other modules use
"""

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
# The .env file should be in the same folder as this file
load_dotenv()

# --- File Upload Settings ---
# Path to the folder where uploaded files are saved
UPLOAD_FOLDER_PATH = "data/uploads"

# --- ChromaDB Settings ---
# Path to the folder where ChromaDB stores its data on disk
CHROMA_DB_PATH = "chroma_db"

# Name of the ChromaDB collection (like a table name in a database)
COLLECTION_NAME = "research_papers"

# --- Supported File Types ---
# List of file extensions that the app can handle
SUPPORTED_EXTENSIONS = ["pdf", "txt", "docx", "xlsx", "csv"]

# --- Embedding Model Settings ---
# Name of the sentence-transformer model used to create embeddings
# all-MiniLM-L6-v2 produces 384-dimensional vectors and is fast
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking Settings ---
# Maximum number of characters in one chunk
CHUNK_SIZE = 800

# Number of overlapping characters between consecutive chunks
# Overlap helps ensure that sentences split across chunk boundaries are still captured
CHUNK_OVERLAP = 100

# --- Loader Grouping Settings ---
# How many lines of a TXT file are grouped into one entry
LINES_PER_GROUP_TXT = 40

# How many paragraphs of a DOCX file are grouped into one entry
PARAGRAPHS_PER_GROUP_DOCX = 10

# How many rows of an Excel or CSV file are grouped into one entry
ROWS_PER_GROUP_TABLE = 20

# --- Retrieval Settings ---
# How many chunks to fetch from ChromaDB in step 1 (broad search)
TOP_K_RETRIEVE = 15

# How many chunks to keep after reranking in step 2 (precise selection)
TOP_K_FINAL = 5

# Cross-encoder model used to rerank the retrieved chunks
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunks scoring below this relevance percentage are hidden from Sources
MIN_RELEVANCE_PERCENT = 20

# --- Groq LLM Settings ---
# The Groq model to use for generating answers
# (llama-3.1-8b-instant was retired on Groq; openai/gpt-oss-20b is the fast free-tier model)
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

# Temperature controls randomness: 0 = deterministic, 1 = very random
# 0.2 is good for factual, accurate answers
LLM_TEMPERATURE = 0.2

# Maximum number of tokens (words/pieces) in the LLM's response
LLM_MAX_TOKENS = 1024

# --- API Key ---
# Read the Groq API key from the .env file
# If the key is not found, GROQ_API_KEY will be None
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# Test block: Run this file alone to check that all settings load correctly
# Usage: python config.py
# ============================================================
if __name__ == "__main__":
    print("=== Config Settings ===")
    print(f"Upload folder: {UPLOAD_FOLDER_PATH}")
    print(f"ChromaDB path: {CHROMA_DB_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"Lines per TXT group: {LINES_PER_GROUP_TXT}")
    print(f"Paragraphs per DOCX group: {PARAGRAPHS_PER_GROUP_DOCX}")
    print(f"Rows per table group: {ROWS_PER_GROUP_TABLE}")
    print(f"Top K retrieve: {TOP_K_RETRIEVE}, Top K final: {TOP_K_FINAL}")
    print(f"Reranker model: {RERANKER_MODEL_NAME}")
    print(f"Min relevance percent: {MIN_RELEVANCE_PERCENT}%")
    print(f"Groq model: {GROQ_MODEL_NAME}")
    print(f"LLM temperature: {LLM_TEMPERATURE}")
    print(f"LLM max tokens: {LLM_MAX_TOKENS}")

    if GROQ_API_KEY:
        # Only show the first 8 characters for security
        print(f"Groq API key: loaded (starts with {GROQ_API_KEY[:8]}...)")
    else:
        print("Groq API key: NOT FOUND - please check your .env file")

    print("\nAll settings loaded successfully!")
