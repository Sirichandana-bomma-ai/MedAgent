import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from utils import load_json

config = load_json("config.json")

class VectorDB:
    def __init__(self):
        self.model = SentenceTransformer(config["embedding_model"])
        self.index = faiss.read_index(config["vector_db_path"])
        with open(f"{config['vector_db_path']}_meta.pkl", "rb") as f:
            self.meta = pickle.load(f)  # (answer, context, source, focus_area)

    def retrieve(self, query, top_k=config["top_k"]):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for i in indices[0]:
            answer, context, source, topic = self.meta[i]
            # Combine context and citation
            results.append(f"{context}\n[Answer: {answer}] [Source: {source}]")
        return results
