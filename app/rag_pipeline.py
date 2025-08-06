import os
from qdrant_client import QdrantClient
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
OLLAMA_INFERENCE_URL = os.getenv("OLLAMA_INFERENCE_URL")
OLLAMA_EMBEDDINGS_URL = os.getenv("OLLAMA_EMBEDDINGS_URL")
OLLAMA_INFERENCE_MODEL = os.getenv("OLLAMA_INFERENCE_MODEL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL")

llm = Ollama(model=OLLAMA_INFERENCE_MODEL, base_url=OLLAMA_INFERENCE_URL)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL, base_url=OLLAMA_EMBEDDINGS_URL)

qdrant_client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
vector_store = Qdrant(client=qdrant_client, embeddings=embeddings, collection_name=QDRANT_COLLECTION_NAME)

retriever = vector_store.as_retriever()

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use at most three sentences and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


def get_rag_response(query: str):
    """
    Runs a query through the RAG pipeline and returns the response.
    """
    retrieved_docs = retriever.get_relevant_documents(query)
    print(f"Retrieved documents for query '{query}': {retrieved_docs}")

    context = {"context": retrieved_docs, "question": query}
    return rag_chain.invoke(context)
