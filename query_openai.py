import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import pickle
import faiss
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_chunks(query, k=5):
    index = faiss.read_index("index/faiss.index")
    with open("index/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

def answer_with_openai(query):
    top_chunks = retrieve_chunks(query)
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])

    prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=300
    )

    answer = response.choices[0].message.content.strip()
    return answer, context