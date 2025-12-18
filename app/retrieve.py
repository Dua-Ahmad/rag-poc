from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os


EMBED_MODEL = "intfloat/multilingual-e5-base"
COLLECTION = "documents"

embedder = SentenceTransformer(EMBED_MODEL)
qdrant = QdrantClient(host="qdrant", port=6333)

def retrieve(query, top_k=5):
    # 1. Embed the query
    query_vector = embedder.encode(
        f"query: information about {query}",
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

       # New-style Qdrant result
        if hasattr(hit, "payload"):
            payload = hit.payload

        # Older / tuple-style fallback
        elif isinstance(hit, tuple):
            for item in hit:
                if isinstance(item, dict):
                    payload = item
                    break

        if not payload:
            continue

        source = payload.get("source", "unknown")
        text = payload.get("text", "")

        if text.strip():
            contexts.append({
                "source": os.path.basename(source),
                "text": text[:MAX_CHARS],
                "score": hit.score,
            })

    # print("QUERY:", query)
    # print("TOP HITS RAW:")
    # for hit in hits:
    #     print(hit)

    return contexts
