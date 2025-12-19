print("🚀 ingest_pptx.py started")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from unstructured.partition.pptx import partition_pptx
import uuid
import os

EMBED_MODEL = "intfloat/multilingual-e5-base"
COLLECTION = "documents"

embedder = SentenceTransformer(EMBED_MODEL)
qdrant = QdrantClient(host="qdrant", port=6333)

if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=embedder.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        )
    )

def ingest_pptx(path, language="en"):
    print(f"📊 Processing PPTX: {path}")
    texts = []

    try:
        elements = partition_pptx(
            filename=path,
            infer_table_structure=True,
        )

        for el in elements:
            if el.text and len(el.text.strip()) > 50:
                texts.append(el.text.strip())

    except Exception as e:
        print(f"❌ PPTX parsing failed: {e}")
        return

    print(f"🧩 Extracted {len(texts)} text blocks")

    if not texts:
        return

    embeddings = embedder.encode(
        [f"passage: {t}" for t in texts],
        normalize_embeddings=True
    )

    points = []
    for text, vector in zip(texts, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": text,
                    "language": language,
                    "source": path,
                    "type": "pptx"
                }
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=points
    )

def ingest_PPTX_folder(folder_path, language="en"):
    print(f"🔍 Scanning folder: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".pptx"):
                full_path = os.path.join(root, filename)
                ingest_pptx(full_path, language)

if __name__ == "__main__":
    ingest_PPTX_folder("/app/data", language="en")
