import faiss
import numpy as np

from app.embedder import load_document,generate_embeddings

def build_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
    "Embeddings must be a non-empty 2D array."
)
    embeddings = np.ascontiguousarray(
        embeddings,dtype=np.float32
    )
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embeddings cannot contain zero vectors.")

    embeddings = embeddings / norms[:, None]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index

def retrieve(index, documents, query_embedding, k=3):
    #return the top-k elements by similarity
    if k < 1:
        raise ValueError("k must be at least 1")
    if not documents:
        return []
    
    query_embedding = np.asarray(
        query_embedding, dtype=np.float32
    ).reshape(1,-1)

    if query_embedding.shape[1] != index.d:
        raise ValueError("Query embedding dimension mismatch")

    k = min(k, index.ntotal)

    scores, indices =  index.search(query_embedding,k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "id": documents[int(idx)]["id"],
            "title": documents[int(idx)]["title"],
            "text": documents[int(idx)]["text"],
            "score": float(score),
        })

    return results

class SemanticSearchEngine:
    def __init__(self):
        self.documents = load_document()
        # Generate embeddings once, when the engine is initialized.
        self.model, embeddings = generate_embeddings(self.documents)
        self.index = build_index(embeddings)

    def search(self, query: str, k: int = 3):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Use the same model used to embed the documents.
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return retrieve(
            self.index,
            self.documents,
            query_embedding[0],
            k=k,
        )


if __name__ == "__main__":
    engine = SemanticSearchEngine()

    while True:
        query = input("\nSearch (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a search query.")
            continue

        for result in engine.search(query, k=3):
            print(f"\n{result['title']} | score={result['score']:.4f}")
            print(result["text"])