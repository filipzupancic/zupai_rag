# 🤖 ZupAI RAG: A Local RAG Microservice

A simple Retrieval-Augmented Generation (RAG) microservice built with **Ollama**, **LangChain**, and **Qdrant**, orchestrated entirely with **Docker Compose**.

For a detailed walkthrough of the architecture and components, check out my full [Medium article](https://medium.com/@filipzupancic123/implement-a-simple-rag-with-ollama-qdrant-and-langchain-cc47beabc290)

## 🏗️ Project Architecture

The system is composed of four main services, running in a single Docker network:

1. **rag-api** 
2. **ollama-inference**
3. **ollama-embeddings** 
4. **qdrant**

## 🚀 Getting Started
Follow these simple steps from your terminal to get the project up and running.

1. 💻 Clone the repository:

```
git clone https://github.com/filipzupancic/zupai_rag.git
cd zupai_rag
```

2. ✅ Set up your environment variables:

```
cp .env.example .env
```

This command copies the example environment file, which contains all the necessary variables for Docker Compose to use.

3. 🛠️ Build and start all services:

```
docker compose up --build -d
```

This single command will build our rag-api Docker image, and then start all four services (ollama-embeddings, ollama-inference, qdrant, and rag-api) in detached mode (-d).

4. 💯 Pull the LLM and Embedding Models:

Next, we'll pull the large language model and the embedding model into their respective Ollama containers.

```
# Pull the LLM for inference (e.g., a larger model)
docker exec ollama-inference ollama pull qwen2.5vl:7b

# Pull a smaller model for embeddings
docker exec ollama-embeddings ollama pull gemma:2b
```

5. 📂 Ingest your documents:

Now that all services are running and the models are downloaded, we can run our ingest_data.py script to populate the Qdrant database.

```
docker compose exec rag-api python /app/ingest_data.py
```

6. 💡 Test the RAG pipeline:

Finally, send a POST request to your API to test the entire system.

```
curl -X POST "http://localhost:8000/prompt" -H "Content-Type: application/json" -d '{ "prompt": "What is RAG?" }'
```

7. 🎯 Your Final Result

If everything is working, you should get a response that is directly based on the content of your document. You can also explore the following links for more debugging and information:

- FastAPI Interactive Docs: http://localhost:8000/docs

- Qdrant Web UI: http://localhost:6333/dashboard

- Application Logs: ```docker logs rag-api```
