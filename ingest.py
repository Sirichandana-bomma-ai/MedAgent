import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from utils import load_json

config = load_json("config.json")

CSV_PATH = config["csv_path"]
VECTOR_DB_PATH = config["vector_db_path"]
EMBEDDING_MODEL = config["embedding_model"]

def ingest_data():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["answer", "context"])  # Ensure context exists

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Combine question + context for better retrieval
    combined_texts = (df['question'] + " " + df['context']).tolist()
    embeddings = model.encode(combined_texts, convert_to_numpy=True)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Store metadata: (answer, context, source, focus_area)
    metadata = list(zip(df['answer'], df['context'], df['source'], df['focus_area']))

    faiss.write_index(index, VECTOR_DB_PATH)
    with open(f"{VECTOR_DB_PATH}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Ingestion completed with context. FAISS index created.")

if __name__ == "__main__":
    ingest_data()
