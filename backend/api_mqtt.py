"""
    Main API file to connect FastAPI with MQTT worker for handling support requests. Frotend and Agents.
"""
import uuid
import json
import asyncio
import time
import sys
from contextlib import asynccontextmanager
from loguru import logger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from monitoring import USER_FEEDBACK
from schemas import ChatRequest, ChatResponse
from mqtt_client import MQTTClient

# 1. Loguru Configuration
logger.remove()
logger.add(
    sys.stdout, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[session_id]}</cyan> - <level>{message}</level>",
    level="INFO",
    enqueue=True 
)
logger = logger.bind(session_id="SYSTEM")

# App Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API Gateway...")

    app.state.main_loop = asyncio.get_running_loop()
    app.state.pending_requests = {}

    try:
        mqtt_client = MQTTClient(broker_host="mosquitto")
        mqtt_client.connect()
        mqtt_client.client.loop_start()
        app.state.mqtt_client = mqtt_client
        logger.success("MQTT Client connected and loop started.")
    except Exception as e:
        logger.critical(f"MQTT connection failed: {e}")
        raise

    listener_task = asyncio.create_task(mqtt_response_listener(app))

    try:
        yield
    finally:
        logger.info("Shutting down API Gateway...")
        listener_task.cancel()
        mqtt_client.client.loop_stop()
        mqtt_client.disconnect()
        logger.success("Cleanup complete.")

# App Init
app = FastAPI(title="Support API 2026", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/feedback")
async def record_feedback(feedback: dict):
    USER_FEEDBACK.labels(rating=feedback["rating"]).inc()
    logger.info(f"Feedback recorded: {feedback['rating']}")
    return {"status": "recorded"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# MQTT Listener
async def mqtt_response_listener(app: FastAPI):
    mqtt_client = app.state.mqtt_client

    def on_message(client, userdata, msg, properties=None):
        try:
            payload = json.loads(msg.payload.decode())
            session_id = payload.get("session_id", "unknown")
            
            # Create a contextual logger tied to this session_id
            request_logger = logger.bind(session_id=session_id)

            if "answer" not in payload and "output" not in payload:
                request_logger.warning("Received invalid MQTT message format (missing 'answer'/'output').")
                return 

            request_logger.info("Gateway received worker response via MQTT.")
            
            future = app.state.pending_requests.pop(session_id, None)
            if future and not future.done():
                app.state.main_loop.call_soon_threadsafe(
                    future.set_result, payload
                )
            else:
                request_logger.error("No pending request found for this session_id (possible timeout/stale).")
        except Exception as e:
            logger.bind(session_id="MQTT-ERROR").exception("MQTT response handling failed internally")

    mqtt_client.client.on_message = on_message
    mqtt_client.client.subscribe("support/responses/+")
    logger.info("Subscribed to support/responses/+")

    while True:
        await asyncio.sleep(1)

# Core Chat Handler
async def handle_chat(request: ChatRequest) -> ChatResponse:
    if request.session_id == "default":
        request.session_id = f"http_{uuid.uuid4().hex[:12]}"

    # Bind the session_id to logs for this specific execution thread
    request_logger = logger.bind(session_id=request.session_id)
    
    future = app.state.main_loop.create_future()
    app.state.pending_requests[request.session_id] = future

    try:
        app.state.mqtt_client.publish(
            f"support/requests/{request.session_id}",
            {
                "question": request.question,
                "session_id": request.session_id,
                "timestamp": time.time(),
            },
        )
        request_logger.info("Chat request published to worker.")
    except Exception as e:
        request_logger.error(f"MQTT publish failed: {e}")
        app.state.pending_requests.pop(request.session_id, None)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Communications broker is unreachable.",
        )

    try:
        response = await asyncio.wait_for(future, timeout=600)
    except asyncio.TimeoutError:
        request_logger.warning("AI Agent timed out.")
        app.state.pending_requests.pop(request.session_id, None)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The AI Agent did not respond in time.",
        )

    request_logger.success("ChatResponse generated successfully.")
    return ChatResponse(
        question=request.question,
        answer=response.get("output") or response.get("answer") or "Error: Agent sent empty response.",
        sources=response.get("sources", []),
        session_id=request.session_id,
        timestamp=response.get("timestamp", time.time()),
    )

# Routes
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await handle_chat(request)

@app.post("/api/v1/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    return await handle_chat(request)

@app.get("/health")
async def health():
    return {"status": "healthy", "time": time.time()}
