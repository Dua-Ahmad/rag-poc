from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os


EMBED_MODEL = "intfloat/multilingual-e5-base"
COLLECTION = "documents"

embedder = SentenceTransformer(EMBED_MODEL)
qdrant = QdrantClient(host="qdrant", port=6333)

from collections import defaultdict

def documents_mentioning(query, top_k=20):
    hits = retrieve(query, top_k=top_k)

    grouped = defaultdict(list)

    for h in hits:
        grouped[h["filename"]].append(h)

    return grouped


def list_documents():
    hits, _ = qdrant.scroll(
        collection_name=COLLECTION,
        with_payload=True,
        limit=1000
    )

    docs = {}

    for p in hits:
        payload = p.payload or {}
        source = payload.get("source")

        if not source:
            continue

        filename = os.path.basename(source)
        ext = os.path.splitext(filename)[1].lower()

        docs[filename] = ext

    return docs

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
                "filename": os.path.basename(source),
                "file_type": os.path.splitext(source)[1].lower(),
            })

    # print("QUERY:", query)
    # print("TOP HITS RAW:")
    # for hit in hits:
    #     print(hit)

    return contexts
