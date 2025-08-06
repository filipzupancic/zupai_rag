import os
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
OLLAMA_EMBEDDINGS_URL = os.getenv("OLLAMA_EMBEDDINGS_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL")


def ingest_documents():
    """
    Loads all documents in the 'documents' directory, splits them,
    creates embeddings, and stores them in Qdrant.
    """
    print("Ingesting documents...")

    loader = DirectoryLoader(path="documents", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL, base_url=OLLAMA_EMBEDDINGS_URL)

    Qdrant.from_documents(
        texts,
        embeddings,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        collection_name=QDRANT_COLLECTION_NAME,
        force_recreate=True
    )
    print(f"Ingested {len(texts)} document chunks.")
    print("Document ingestion complete.")


if __name__ == "__main__":
    os.makedirs("documents", exist_ok=True)

    rag_content = ("This is a simple document about Retrieval-Augmented Generation (RAG). "
                   "RAG is a technique that combines a retriever (like a vector database) "
                   "and a large language model (LLM) to generate more accurate and "
                   "context-aware responses. It first retrieves relevant information "
                   "from a knowledge base and then uses that information as context "
                   "for the LLM to generate a response.")
    with open("documents/rag_definition.txt", "w") as f:
        f.write(rag_content)

    filip_content = ("John Doe was a famous poet and writer who was loved by many and hated by few.")
    with open("documents/filip_info.txt", "w") as f:
        f.write(filip_content)

    ingest_documents()
