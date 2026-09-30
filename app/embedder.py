import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).resolve().parent.parent/"data"/"documents.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_document():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

def generate_embeddings(documents):
    model  = SentenceTransformer(MODEL_NAME)

    texts = [
        f"{doc['title']}.{doc['text']}"
        for doc in documents
    ]

    embeddings  = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return model, np.asarray(embeddings,dtype=np.float32)

if __name__ == '__main__':
    documents = load_document()
    models, embeddings = generate_embeddings(documents)

    print("Number of documents: ", len(documents))
    print("Embedding shape: ", embeddings.shape)
    print("Embedding dtype: ", embeddings.dtype)
    print("First embedding: ", embeddings[0][:0])