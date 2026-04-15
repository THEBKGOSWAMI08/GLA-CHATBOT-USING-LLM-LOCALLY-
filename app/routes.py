import traceback
from flask import Blueprint, request, jsonify, render_template, current_app

bp = Blueprint("main", __name__)


@bp.route("/")
def home():
    return render_template("index.html")


@bp.route("/chat", methods=["POST"])
def chat():
    rag_chain = current_app.rag_chain

    if not rag_chain:
        return jsonify({"answer": "Error: RAG system is not initialized. Please check server logs."})

    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"answer": "Please ask a question."})

        print(f"\n📝 User: {question}")
        answer = rag_chain.invoke(question)
        print(f"🤖 Bot: {answer}")

        return jsonify({"answer": answer})

    except Exception as e:
        print("Backend Error:", e)
        traceback.print_exc()
        return jsonify({"answer": "Something went wrong at backend."})
