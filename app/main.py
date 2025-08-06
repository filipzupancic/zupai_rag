from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import get_rag_response


app = FastAPI(
    title="Local RAG Microservice",
    description="A simple RAG service using Ollama, LangChain, and Qdrant."
)

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    """
    A simple root endpoint to check if the service is running.
    """
    return {"message": "Welcome to the RAG Microservice! Use the /prompt endpoint to ask questions."}

@app.post("/prompt")
def handle_prompt(request: PromptRequest):
    """
    Accepts a prompt, runs it through the RAG pipeline, and returns the response.
    """
    try:
        response = get_rag_response(request.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
