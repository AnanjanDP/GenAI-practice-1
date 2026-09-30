
import numpy as np
import pytest
import faiss

from app.search import build_index, retrieve


@pytest.fixture
def sample_data():
    documents = [
        {"id": 1, "title": "Authentication", "text": "Secure accounts"},
        {"id": 2, "title": "Databases", "text": "Optimize SQL queries"},
        {"id": 3, "title": "Caching", "text": "Store frequently used data"},
    ]

    # Unit vectors pointing in different directions.
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return documents, embeddings


def test_build_index(sample_data):
    documents, embeddings = sample_data

    index = build_index(embeddings)

    assert index.ntotal == len(documents)
    assert index.d == 3


def test_retrieve_returns_most_similar_first(sample_data):
    documents, embeddings = sample_data
    index = build_index(embeddings)

    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    query /= np.linalg.norm(query)

    results = retrieve(index, documents, query, k=2)

    assert len(results) == 2
    assert results[0]["title"] == "Authentication"
    assert results[0]["score"] > results[1]["score"]


def test_k_cannot_exceed_document_count(sample_data):
    documents, embeddings = sample_data
    index = build_index(embeddings)

    results = retrieve(
        index, documents, embeddings[0], k=10
    )

    assert len(results) == 3


def test_invalid_k_raises_error(sample_data):
    documents, embeddings = sample_data
    index = build_index(embeddings)

    with pytest.raises(ValueError, match="k must be"):
        retrieve(index, documents, embeddings[0], k=0)


def test_empty_documents_return_empty_results():
    index = faiss.IndexFlatIP(3)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert retrieve(index, [], query, k=3) == []


def test_wrong_embedding_dimension_raises_error(sample_data):
    documents, embeddings = sample_data
    index = build_index(embeddings)

    with pytest.raises(ValueError, match="dimension mismatch"):
        retrieve(index, documents, np.array([1.0, 0.0]), k=2)


def test_empty_embeddings_are_rejected():
    with pytest.raises(ValueError, match="non-empty 2D"):
        build_index(np.empty((0, 384), dtype=np.float32))