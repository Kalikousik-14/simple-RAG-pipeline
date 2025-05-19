import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def build_faiss_index(chunks):

    os.makedirs("index", exist_ok=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "index/faiss.index")
    with open("index/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks