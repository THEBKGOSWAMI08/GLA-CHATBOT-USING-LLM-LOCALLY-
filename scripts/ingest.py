"""
Ingestion script — builds the vector database from documents in data/.

Called automatically by the app on startup when new files are detected.
Can also be run manually:
    python scripts/ingest.py
"""

import os
import sys
import glob

# Ensure project root is on the path when run as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from config import DB_DIR, DATA_DIR, OLLAMA_EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def ingest():
    print("🚀 Starting Ingestion Process...")
    print(f"   Data directory : {DATA_DIR}")
    print(f"   Vector DB path : {DB_DIR}")

    docs = []

    # PDF files
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if pdf_files:
        print(f"\n📄 Found {len(pdf_files)} PDF file(s):")
        for path in pdf_files:
            try:
                print(f"   Processing: {os.path.basename(path)}...")
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
                print(f"   ✅ Loaded {os.path.basename(path)}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
    else:
        print("\n⚠️  No PDF files found in data/.")

    # Markdown files
    md_files = glob.glob(os.path.join(DATA_DIR, "*.md"))
    if md_files:
        print(f"\n📝 Found {len(md_files)} Markdown file(s):")
        for path in md_files:
            try:
                print(f"   Processing: {os.path.basename(path)}...")
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
                print(f"   ✅ Loaded {os.path.basename(path)}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")

    # Text files
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if txt_files:
        print(f"\n📃 Found {len(txt_files)} Text file(s):")
        for path in txt_files:
            try:
                print(f"   Processing: {os.path.basename(path)}...")
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
                print(f"   ✅ Loaded {os.path.basename(path)}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")

    if not docs:
        print("\n❌ No documents loaded. Add .md, .txt, or .pdf files to data/ and retry.")
        return

    print(f"\n✂️  Splitting {len(docs)} document(s) into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(docs)
    print(f"   ✅ Created {len(splits)} chunks.")

    print("\n🧠 Generating embeddings and saving to vector DB...")
    Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model=OLLAMA_EMBED_MODEL),
        persist_directory=DB_DIR,
    )

    print(f"\n✅ Vector database saved to '{DB_DIR}'")
    print("🎉 Ingestion complete!")


if __name__ == "__main__":
    ingest()
