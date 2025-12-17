
# RAG Pipeline with Docker - with GPU usage
This project implements a Retrieval-Augmented Generation pipeline using:
- Qdrant for vector storage
- Sentence Transformers for embeddings
- Unstructured for PDF parsing
- Docker Compose for orchestration
- **Ollama** for local, open-source LLM inference. **Model:** `qwen2.5:1.5b`


## How to run
1. docker compose up -d --build
2. docker exec -it rag-app bash
3. python /app/ingest.py
"
