from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os


EMBED_MODEL = "intfloat/multilingual-e5-base"
COLLECTION = "documents"

embedder = SentenceTransformer(EMBED_MODEL)
qdrant = QdrantClient(host="qdrant", port=6333)

def retrieve(query, top_k=2):
    # 1. Embed the query
    query_vector = embedder.encode(
        f"query: {query}",
        normalize_embeddings=True
    ).tolist()

    # 2. Query Qdrant (new-style API)
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    # 3. Extract text payloads
    MAX_CHARS = 300
    contexts = []

    for hit in hits.points:
        payload = None

        # Case 1: new-style object
        if hasattr(hit, "payload"):
            payload = hit.payload

        # Case 2: tuple return (varies by version)
        elif isinstance(hit, tuple):
            for item in hit:
                if isinstance(item, dict):
                    payload = item
                    break

        if payload and "text" in payload:
            contexts.append(payload["text"][:MAX_CHARS])

    return contexts
