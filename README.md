
echo "# RAG Pipeline with Docker - using GPU
This project implements a Retrieval-Augmented Generation pipeline using:
- Qdrant for vector storage
- Sentence Transformers for embeddings
- Unstructured for PDF parsing
- Docker Compose for orchestration

## How to run
1. docker compose up -d --build
2. docker exec -it rag-app bash
3. python /app/ingest.py
"
