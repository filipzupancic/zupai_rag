# zupai_rag

- git clone https://github.com/filipzupancic/zupai_rag.git

- cd zupai_rag

- cp .env.example .env

- docker compose up --build -d

- docker exec ollama-inference ollama pull qwen2.5vl:7b

- docker exec ollama-embeddings ollama pull gemma:2b

- docker compose exec rag-api python /app/ingest_data.py 

- curl -X POST "http://localhost:8000/prompt" -H "Content-Type: application/json" -d '{ "prompt": "What is RAG?" }'

- Swagger: http://localhost:8000/docs

- Logs: docker logs rag-api


