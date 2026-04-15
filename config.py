import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DB_DIR = os.path.join(BASE_DIR, "vector_db")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ollama models
OLLAMA_LLM_MODEL = "llama3.2:3b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Ingestion
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Server
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True
