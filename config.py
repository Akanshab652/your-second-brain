import os

# Gemini API key (set as environment variable)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Local data file (notes, transcripts, etc.)
DATA_FILE = "data/notes.txt"

# Embedding model name for Llama
LLAMA_EMBED_MODEL = "llama-embedding-model"

# Number of chunks to return
TOP_K = 3
