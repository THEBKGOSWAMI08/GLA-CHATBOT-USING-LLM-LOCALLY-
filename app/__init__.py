import os
import glob
from flask import Flask
from flask_cors import CORS

from config import DB_DIR, DATA_DIR
from .rag import create_chain


def _needs_ingest():
    """Return True if the vector DB is missing or any data file is newer than the DB."""
    chroma_db = os.path.join(DB_DIR, "chroma.sqlite3")

    if not os.path.exists(chroma_db):
        return True

    db_mtime = os.path.getmtime(chroma_db)

    data_files = (
        glob.glob(os.path.join(DATA_DIR, "*.md"))
        + glob.glob(os.path.join(DATA_DIR, "*.txt"))
        + glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    )

    if not data_files:
        return False

    return max(os.path.getmtime(f) for f in data_files) > db_mtime


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
    )
    CORS(app)

    print("--------------------------------------------------")

    if _needs_ingest():
        print("📂 New or missing data detected — running ingestion...")
        print("--------------------------------------------------")
        from scripts.ingest import ingest
        ingest()
        print("--------------------------------------------------")

    print("⏳ INITIALIZING RAG PIPELINE...")
    print("--------------------------------------------------")

    app.rag_chain = create_chain()

    if app.rag_chain:
        print("✅ RAG PIPELINE LOADED SUCCESSFULLY!")
    else:
        print("❌ RAG PIPELINE FAILED TO LOAD.")

    print("--------------------------------------------------")

    from .routes import bp
    app.register_blueprint(bp)

    return app
