from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from unstructured.partition.pdf import partition_pdf
from pypdf import PdfReader
import uuid
import os

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
    print(f"Processing: {path}")
    texts = []

    # ---- Try normal PDF extraction first ----
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                texts.append(text.strip())
    except Exception as e:
        print(f"pypdf failed: {e}")

   # ---- Fallback to unstructured (scanned PDFs) ----
    if not texts:
        print("Falling back to unstructured PDF parsing...")
        elements = partition_pdf(
            filename=path,
            strategy="auto",          # auto-detect text vs scanned
            infer_table_structure=True,
        )
        for el in elements:
            if el.text and len(el.text.strip()) > 50:
                texts.append(el.text.strip())

    # If still no text, skip
    if not texts:
        print("No text extracted, skipping.")
        return
    
    # Embed and upsert to Qdrant
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


# ---------- FOLDER ingestion ----------
def ingest_PDF_folder(folder_path, language="en"):
    print(f"🔍 Scanning folder recursively: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(root, filename)
                print(f"Ingesting: {full_path}")
                ingest_pdf(full_path, language=language)

if __name__ == "__main__":
    ingest_PDF_folder("/app/data", language="en")

