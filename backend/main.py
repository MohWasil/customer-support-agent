from fastapi import FastAPI
from dotenv import load_dotenv
from rag_simple import SimpleRAG
import os

# Load environment variables
load_dotenv()

app = FastAPI(title="Customer Support Agent API")

@app.get("/")
def read_root():
    return {"message": "Customer Support Agent is running"}

# Initialize RAG at startup
rag = SimpleRAG("./data/knowledge_base")

@app.post("/api/v1/chat")
def chat_endpoint(question: dict):
    q = question.get("question")
    result = rag.query(q)
    return {
        "question": q,
        "answer": result["answer"],
        "sources": result["sources"],
        "session_id": "test_123"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": os.getenv("OLLAMA_MODEL")}
