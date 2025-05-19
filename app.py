from flask import Flask, request, jsonify
import os
from ingest import ingest_file, chunk_text
from embed import build_faiss_index
from model_loader import load_local_llm
from query_local import retrieve_chunks as retrieve_local
from query_openai import answer_with_openai

app = Flask(__name__)
UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

llm = load_local_llm()

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        raw_text = ingest_file(save_path)
        chunks = chunk_text(raw_text)
        build_faiss_index(chunks)
        return jsonify({
            "message": f"Successfully ingested and indexed {filename}",
            "num_chunks": len(chunks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask_local", methods=["POST"])
def ask_local():
    data = request.json
    query = data.get("question", "")
    if not query:
        return jsonify({"error": "No question provided."}), 400

    top_chunks = retrieve_local(query)
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])
    prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""
    output = llm(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]

    return jsonify({
        "question": query,
        "answer": output.strip(),
        "context": context
    })

@app.route("/ask_openai", methods=["POST"])
def ask_openai():
    data = request.json
    query = data.get("question", "")
    if not query:
        return jsonify({"error": "No question provided."}), 400

    answer, context = answer_with_openai(query)
    return jsonify({
        "question": query,
        "answer": answer,
        "context": context
    })

if __name__ == "__main__":
    app.run(debug=True)