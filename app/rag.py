import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import DB_DIR, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def create_chain():
    print(f"🔄 Loading Vector Database from '{DB_DIR}'...")

    if not os.path.exists(DB_DIR):
        print(f"❌ Error: '{DB_DIR}' not found. Run 'python scripts/ingest.py' first.")
        return None

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=OllamaEmbeddings(model=OLLAMA_EMBED_MODEL),
    )
    retriever = vectorstore.as_retriever()
    print("✅ Vector Database Loaded Successfully.")

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for answering user questions based on the retrieved context provided below.

Follow these rules:
- Use only the information from the context. Do not make up answers.
- If the answer is not clearly in the context, reply: "I don't know based on the given context."
- Keep the answer brief (3–4 sentences max).
- Prefer clarity over long explanations.

Context:
{context}

Question:
{question}

Answer:
""")

    llm = ChatOllama(model=OLLAMA_LLM_MODEL)

    def log_retrieved_docs(docs):
        print(f"\n🔍 RETRIEVED {len(docs)} DOCS:")
        for i, doc in enumerate(docs):
            print(f"   [{i+1}] {doc.page_content[:150]}...")
        return docs

    rag_chain = (
        {
            "context": retriever | log_retrieved_docs | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
