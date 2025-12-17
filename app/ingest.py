from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pypdf import PdfReader
import uuid

EMBED_MODEL = "intfloat/multilingual-e5-base"
COLLECTION = "documents"

# Load embedding model
embedder = SentenceTransformer(EMBED_MODEL)

# Qdrant client
qdrant = QdrantClient(host="qdrant", port=6333)

# Create collection if not exists
if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=embedder.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        )
    )

def ingest_pdf(path, language="en"):
    reader = PdfReader(path)

    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)

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
                    "source": path
                }
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=points
    )

if __name__ == "__main__":
    ingest_pdf("/app/PNG_Strategy_to_Prevent_GBV.pdf", language="en")
