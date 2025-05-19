import pickle
from sentence_transformers import SentenceTransformer
import faiss

def retrieve_chunks(user_query, k=5):
    
    index = faiss.read_index("index/faiss.index")
    with open("index/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = embedder.encode([user_query])
    D, I = index.search(query_vec, k)

    return [chunks[i] for i in I[0]]