# from fastapi import FastAPI
# from dotenv import load_dotenv
# from rag_simple import SimpleRAG
# import os

# # Load environment variables
# load_dotenv()

# app = FastAPI(title="Customer Support Agent API")

# @app.get("/")
# def read_root():
#     return {"message": "Customer Support Agent is running"}

# # Initialize RAG at startup
# rag = SimpleRAG("./data/knowledge_base")

# @app.post("/api/v1/chat")
# def chat_endpoint(question: dict):
#     q = question.get("question")
#     result = rag.query(q)
#     return {
#         "question": q,
#         "answer": result["answer"],
#         "sources": result["sources"],
#         "session_id": "test_123"
#     }

# @app.get("/health")
# def health_check():
#     return {"status": "healthy", "model": os.getenv("OLLAMA_MODEL")}




# backend/main.py - final version for Week 2
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer
from schemas import ChatRequest, ChatResponse
from agent import SupportAgent
import time

app = FastAPI(
    title="Customer Support Agent API",
    description="AI-powered customer support with RAG and ReAct agent",
    version="1.0.0"
)

# Security
security = HTTPBearer()  # For future auth
agent = SupportAgent()

# Rate limiting (simple in-memory)
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_calls=10, window=60):
        self.calls = defaultdict(list)
        self.max_calls = max_calls
        self.window = window
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        self.calls[client_id] = [t for t in self.calls[client_id] if now - t < self.window]
        
        if len(self.calls[client_id]) >= self.max_calls:
            return False
        
        self.calls[client_id].append(now)
        return True

rate_limiter = RateLimiter()

def get_client_id(request: Request):
    # Use IP as client identifier
    # Check if request.client is None before accessing its attributes
    if request.client is None:
        # Return a default identifier for testing or missing client info
        # You could also raise an exception if client IP is strictly required.
        # Returning a fixed string like "test_client" is common in tests.
        return "test_client" 
    else:
        return request.client.host

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    client_id: str = Depends(get_client_id)
):
    # Rate limiting
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    try:
        # Run agent
        result = agent.run(request.question)
        
        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            sources=[],  # We'll add sources next week
            session_id=request.session_id,
            timestamp=time.time()
        )
        
    except Exception as e:
        # Log the error internally
        print(f"ERROR: {e}")
        # Return generic message to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process request at this time."
        )

@app.get("/health")
def health_check():
    return {
        "status": "operational",
        "components": {
            "agent": "ready",
            "rate_limiter": "active"
        }
    }
