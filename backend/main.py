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

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount frontend static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return FileResponse("../frontend/index.html")
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



















# import json
# import time
# import logging
# from agent import SupportAgent
# from mqtt_client import MQTTClient

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("agent-worker")

# agent = SupportAgent()
# mqtt = MQTTClient()

# def on_request(client, userdata, msg, properties=None):
#     try:
#         payload = json.loads(msg.payload.decode())
#         question = payload.get("question")
#         session_id = payload.get("session_id")

#         logger.info(f"Processing request: {session_id}")

#         result = agent.run(question)

#         response = {
#             "session_id": session_id,
#             "output": result.get("answer"),
#             "timestamp": time.time(),
#             "sources": []
#         }

#         mqtt.publish(
#             f"support/responses/{session_id}",
#             response
#         )

#     except Exception as e:
#         logger.exception("Agent processing failed")

# def main():
#     mqtt.connect()
#     mqtt.client.on_message = on_request
#     mqtt.client.subscribe("support/requests/+")
#     mqtt.client.loop_forever()

# if __name__ == "__main__":
#     main()
