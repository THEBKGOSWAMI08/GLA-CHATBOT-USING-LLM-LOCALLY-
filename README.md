# GLA University AI Chatbot

A Retrieval-Augmented Generation (RAG) based AI chatbot for GLA University, Mathura. Answers student and visitor queries about admissions, academics, fees, placements, campus facilities, and more — powered by a locally hosted LLM via Ollama. No external API keys required.

---

## Project Structure

```
GLA-CHATBOT-USING-LLM-LOCALLY/
│
├── run.py                     # Entry point — python run.py
├── config.py                  # All configuration constants
├── pyproject.toml             # Poetry dependency spec
│
├── app/                       # Flask application package
│   ├── __init__.py            # App factory + auto-ingest on startup
│   ├── routes.py              # Route handlers (/, /chat)
│   └── rag.py                 # RAG chain builder
│
├── scripts/
│   └── ingest.py              # Ingestion logic (auto-run or manual)
│
├── data/                      # Knowledge base — drop files here
│   └── GLA_University_Data.md
│
├── vector_db/                 # Persisted ChromaDB (auto-generated, gitignored)
│
├── templates/
│   └── index.html             # Chat UI
│
└── static/
    └── image/
        └── gla-logo.jpg
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Web Framework | Flask, Flask-CORS |
| RAG Orchestration | LangChain, LangChain-Community |
| LLM | Ollama (`llama3.2:3b`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector Database | ChromaDB |
| Document Parsing | PyPDF |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- [Ollama](https://ollama.com) installed and running

### 1. Pull Required Ollama Models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 2. Install Python Dependencies

```bash
poetry install
```

### 3. Start the Server

```bash
poetry run python run.py
```

Or activate the Poetry shell first, then run the server:

```bash
poetry shell
python run.py
```

That's it. On startup the app will:
- Automatically ingest all files in `data/` if the vector DB is missing
- Automatically re-ingest if any file in `data/` is newer than the existing DB
- Skip ingestion and load instantly if nothing has changed

The chatbot will be available at [http://localhost:5000](http://localhost:5000).

---

## Adding or Updating Knowledge

Drop any `.md`, `.txt`, or `.pdf` file into the `data/` directory and restart the server. Ingestion runs automatically.

```
data/
├── GLA_University_Data.md     ← existing
└── new_topic.pdf              ← just added — will be picked up on next run
```

```bash
poetry run python run.py   # detects new file, re-ingests, then starts
```

---

## How It Works

### RAG Pipeline (Retrieval-Augmented Generation)

```
User Question
     │
     ▼
Embedding Model  ──────►  Vector Database
(encode question)         (semantic search)
                               │
                               ▼
                        Relevant Context
                               │
                               ▼
                    Prompt = Question + Context
                               │
                               ▼
                       LLM (llama3.2:3b)
                               │
                               ▼
                           Answer
```

1. **Ingestion (automatic):** Documents in `data/` are chunked, embedded, and stored in ChromaDB. Runs on startup only when needed.
2. **Retrieval:** The user's question is embedded and matched against stored chunks using cosine similarity.
3. **Generation:** Top-matching context is injected into a prompt and sent to the local LLM, which produces a grounded answer.

### Supported Topics

- Admissions process and eligibility
- Available programs (B.Tech, MBA, BBA, BCA, MCA, B.Pharm, and more)
- Tuition fees and scholarships
- Placement statistics and top recruiters
- Campus facilities (hostels, labs, library, sports)
- Campus life (clubs, events, food)
- Contact and location details

---

## API Reference

**`POST /chat`**

Request:
```json
{
  "question": "What is the fee structure for B.Tech?"
}
```

Response:
```json
{
  "answer": "The B.Tech fee at GLA University is approximately ..."
}
```

---

## Configuration

All constants are in [config.py](config.py).

| Parameter | Default | Description |
|---|---|---|
| `OLLAMA_LLM_MODEL` | `llama3.2:3b` | LLM used for generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `DB_DIR` | `./vector_db` | ChromaDB persistence path |
| `DATA_DIR` | `./data` | Directory scanned for source documents |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `FLASK_HOST` | `127.0.0.1` | Server host |
| `FLASK_PORT` | `5000` | Server port |

---

## Manual Ingestion

Ingestion runs automatically, but you can also trigger it manually:

```bash
poetry run python scripts/ingest.py
```

---

## Troubleshooting

**Ollama connection refused**
Ensure Ollama is running: `ollama serve`. Verify models are available: `ollama list`.

**No documents loaded**
Ensure at least one `.md`, `.txt`, or `.pdf` file exists in `data/`.

**Stale answers after updating data**
Restart the server — it will detect the updated files and re-ingest automatically.

---

## License

This project was developed for GLA University, Mathura, India. All university data used in the knowledge base is publicly available information.
