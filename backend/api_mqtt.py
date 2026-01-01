# # backend/api_mqtt.py
# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from schemas import ChatRequest, ChatResponse
# from mqtt_client import MQTTClient
# import uuid
# import json
# import asyncio
# import time

# app = FastAPI()
# mqtt_client = MQTTClient()

# # Store pending requests (session_id -> Future)
# pending_requests = {}

# @app.on_event("startup")
# async def startup_event():
#     """Connect to MQTT on API startup"""
#     mqtt_client.connect()
#     # Start background task to process responses
#     asyncio.create_task(mqtt_response_listener())

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Disconnect MQTT gracefully"""
#     mqtt_client.disconnect()

# async def mqtt_response_listener():
#     """Listen for MQTT responses and fulfill pending requests"""
#     import paho.mqtt.client as mqtt
    
#     def on_response(client, userdata, msg):
#         try:
#             payload = json.loads(msg.payload.decode())
#             session_id = payload.get("session_id")
            
#             # Fulfill the pending request
#             if session_id in pending_requests:
#                 future = pending_requests.pop(session_id)
#                 future.set_result(payload)
                
#         except Exception as e:
#             print(f"Response listener error: {e}")
    
#     # Subscribe to all response topics
#     mqtt_client.client.on_message = on_response
#     mqtt_client.client.subscribe("support/responses/+")

# @app.post("/api/v1/chat", response_model=ChatResponse)
# async def chat_via_mqtt(request: ChatRequest, background_tasks: BackgroundTasks):
#     """
#     Async chat endpoint using MQTT:
#     1. Publishes question to MQTT
#     2. Waits for response (with timeout)
#     3. Returns answer
#     """
#     # Generate unique session ID if not provided
#     if request.session_id == "default":
#         request.session_id = f"http_{uuid.uuid4().hex[:12]}"
    
#     # Create future for async waiting
#     loop = asyncio.get_event_loop()
#     future = loop.create_future()
#     pending_requests[request.session_id] = future
    
#     # Publish to MQTT
#     mqtt_client.publish(
#         f"support/requests/{request.session_id}",
#         {
#             "question": request.question,
#             "session_id": request.session_id,
#             "timestamp": time.time()
#         }
#     )
    
#     # Wait for response (timeout after 30 seconds)
#     try:
#         response = await asyncio.wait_for(future, timeout=30.0)
        
#         return ChatResponse(
#             question=request.question,
#             answer=response["answer"],
#             sources=[],  # We'll add source tracking next week
#             session_id=request.session_id,
#             timestamp=response["timestamp"]
#         )
        
#     except asyncio.TimeoutError:
#         pending_requests.pop(request.session_id, None)
#         raise HTTPException(
#             status_code=504,
#             detail="Request timed out. Agent may be overloaded."
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to process request"
#         )

# # Quick test endpoint
# @app.post("/api/v1/chat/sync")
# def chat_sync(request: ChatRequest):
#     """Synchronous endpoint for easy testing"""
#     from agent import SupportAgent
#     agent = SupportAgent()
#     result = agent.run(request.question)
    
#     return ChatResponse(
#         question=request.question,
#         answer=result["answer"],
#         sources=[],
#         session_id=request.session_id,
#         timestamp=time.time()
#     )









# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from schemas import ChatRequest, ChatResponse
# from mqtt_client import MQTTClient
# import uuid
# import json
# import asyncio
# import time

# # Import Lifespan from fastapi
# from contextlib import asynccontextmanager


# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # Add this block:
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"], # Allow your frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Define the lifespan function
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Define startup and shutdown logic."""
#     # Startup
#     print("Starting up...")
#     mqtt_client = MQTTClient() # Create MQTTClient instance
#     mqtt_client.connect()
#     # Store the client instance in the app state for use in endpoints if needed
#     # Or keep it as a global variable if preferred, though state is cleaner
#     app.state.mqtt_client = mqtt_client 
#     # Start background task to process responses
#     # Note: asyncio.create_task might not work as expected in lifespan
#     # It's better to manage background tasks separately or use FastAPI's background tasks for short-lived ones.
#     # For a persistent listener, consider running it in the main thread or a dedicated thread/process if necessary.
#     # For this example, we'll assume mqtt_response_listener runs in a background thread managed by the MQTT client itself,
#     # or we start it here but ensure it's handled correctly.
#     # Let's start the listener loop within the lifespan, ensuring it stops on shutdown.
#     response_listener_task = asyncio.create_task(mqtt_response_listener(mqtt_client))
    
#     yield # The application runs from this point onwards

#     # Shutdown
#     print("Shutting down...")
#     # Stop the listener task
#     response_listener_task.cancel()
#     try:
#         await response_listener_task
#     except asyncio.CancelledError:
#         pass # Expected after cancellation
#     # Disconnect MQTT
#     mqtt_client.disconnect()

# # Initialize FastAPI app with the lifespan function
# app = FastAPI(lifespan=lifespan)

# # Store pending requests (session_id -> Future)
# pending_requests = {}

# # The mqtt_response_listener needs to be updated to accept the client instance
# async def mqtt_response_listener(mqtt_client_instance: MQTTClient):
#     """Listen for MQTT responses and fulfill pending requests"""
#     import paho.mqtt.client as mqtt
    
#     def on_response(client, userdata, msg):
#         global pending_requests # Access the outer scope's pending_requests dict
#         try:
#             payload = json.loads(msg.payload.decode())
#             session_id = payload.get("session_id")
            
#             # Fulfill the pending request
#             if session_id in pending_requests:
#                 future = pending_requests.pop(session_id)
#                 if not future.done(): # Check if future wasn't already cancelled/fulfilled
#                     future.set_result(payload)
#                 else:
#                     print(f"Future for session {session_id} was already done when response arrived.")
                
#         except Exception as e:
#             print(f"Response listener error: {e}")
#             import traceback
#             traceback.print_exc()
    
#     # Subscribe to all response topics using the passed client instance
#     mqtt_client_instance.client.on_message = on_response
#     mqtt_client_instance.client.subscribe("support/responses/+")
    
#     # Keep this coroutine alive to maintain the listener
#     # This is tricky because on_message is a callback from the MQTT client's loop,
#     # which runs in a different thread (started by loop_start in mqtt_client).
#     # The task running this coroutine might not be necessary just for the callbacks to happen,
#     # as the MQTT client handles its own network loop in the background thread.
#     # The important part is that the MQTT client stays connected and the loop runs.
#     # We could add a simple sleep loop here to keep the task alive, but it's not actively doing work.
#     # The cancellation in lifespan should handle stopping this task gracefully.
#     # For now, let's just have it sleep, assuming the MQTT client's own loop handles messages.
#     try:
#         while True:
#             await asyncio.sleep(60) # Sleep indefinitely, but allow cancellation
#     except asyncio.CancelledError:
#         print("mqtt_response_listener task cancelled.")
#         # Optionally unsubscribe here if needed, but disconnect should handle it
#         # mqtt_client_instance.client.unsubscribe("support/responses/+")
#         raise # Re-raise to confirm cancellation

# @app.post("/api/v1/chat", response_model=ChatResponse)
# async def chat_via_mqtt(request: ChatRequest, background_tasks: BackgroundTasks):
#     """
#     Async chat endpoint using MQTT:
#     1. Publishes question to MQTT
#     2. Waits for response (with timeout)
#     3. Returns answer
#     """
#     global pending_requests # Access the global dict
    
#     # Generate unique session ID if not provided
#     if request.session_id == "default":
#         request.session_id = f"http_{uuid.uuid4().hex[:12]}"
    
#     # Create future for async waiting
#     loop = asyncio.get_event_loop()
#     future = loop.create_future()
#     pending_requests[request.session_id] = future
    
#     # Access the MQTT client from app state
#     mqtt_client_instance = app.state.mqtt_client
#     # Publish to MQTT
#     mqtt_client_instance.publish(
#         f"support/requests/{request.session_id}",
#         {
#             "question": request.question,
#             "session_id": request.session_id,
#             "timestamp": time.time()
#         }
#     )
    
#     # Wait for response (timeout after 30 seconds)
#     try:
#         response = await asyncio.wait_for(future, timeout=30.0)
        
#         return ChatResponse(
#             question=request.question,
#             answer=response["answer"],
#             sources=[],  # We'll add source tracking next week
#             session_id=request.session_id,
#             timestamp=response["timestamp"]
#         )
        
#     except asyncio.TimeoutError:
#         # Remove the future from pending_requests if it hasn't been fulfilled
#         # Use pop with default to avoid KeyError if already removed by callback
#         pending_requests.pop(request.session_id, None) 
#         raise HTTPException(
#             status_code=504,
#             detail="Request timed out. Agent may be overloaded."
#         )
#     except Exception as e:
#         # Also remove on other exceptions
#         pending_requests.pop(request.session_id, None)
#         print(f"Error in chat_via_mqtt: {e}") # Log the error
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to process request"
#         )

# # Quick test endpoint
# @app.post("/api/v1/chat/sync")
# def chat_sync(request: ChatRequest):
#     """Synchronous endpoint for easy testing"""
#     from agent import SupportAgent
#     agent = SupportAgent()
#     result = agent.run(request.question)
    
#     return ChatResponse(
#         question=request.question,
#         answer=result["answer"],
#         sources=[],
#         session_id=request.session_id,
#         timestamp=time.time()
#     )





















# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from schemas import ChatRequest, ChatResponse
# from mqtt_client import MQTTClient
# import uuid
# import json
# import asyncio
# import time
# from contextlib import asynccontextmanager
# from fastapi.middleware.cors import CORSMiddleware

# # 1. Define the lifespan function BEFORE initializing the app
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Define startup and shutdown logic."""
#     # Startup
#     print("Starting up...")
#     mqtt_client = MQTTClient() 
#     mqtt_client.connect()
#     mqtt_client.client.loop_start() 
#     # Store the client instance in the app state
#     app.state.mqtt_client = mqtt_client 
    
#     # Start background task to process responses
#     response_listener_task = asyncio.create_task(mqtt_response_listener(mqtt_client))
    
#     yield # The application runs here

#     # Shutdown
#     print("Shutting down...")
#     response_listener_task.cancel()
#     try:
#         await response_listener_task
#     except asyncio.CancelledError:
#         pass 
#     mqtt_client.disconnect()

# # 2. Initialize FastAPI app ONCE with lifespan and CORS
# app = FastAPI(lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows any origin for testing; change to ["http://localhost:3000"] for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Store pending requests (session_id -> Future)
# pending_requests = {}

# async def mqtt_response_listener(mqtt_client_instance: MQTTClient):
#     """Listen for MQTT responses and fulfill pending requests"""
#     import paho.mqtt.client as mqtt
    
#     def on_response(client, userdata, msg):
#         global pending_requests 
#         try:
#             payload = json.loads(msg.payload.decode())
#             session_id = payload.get("session_id")
            
#             if session_id in pending_requests:
#                 future = pending_requests.pop(session_id)
#                 if not future.done():
#                     future.set_result(payload)
                
#         except Exception as e:
#             print(f"Response listener error: {e}")
    
#     mqtt_client_instance.client.on_message = on_response
#     mqtt_client_instance.client.subscribe("support/responses/+")
    
#     try:
#         while True:
#             await asyncio.sleep(60) 
#     except asyncio.CancelledError:
#         print("mqtt_response_listener task cancelled.")
#         raise

# @app.post("/api/v1/chat", response_model=ChatResponse)
# async def chat_via_mqtt(request: ChatRequest, background_tasks: BackgroundTasks):
#     global pending_requests 
    
#     if request.session_id == "default":
#         request.session_id = f"http_{uuid.uuid4().hex[:12]}"
    
#     loop = asyncio.get_event_loop()
#     future = loop.create_future()
#     pending_requests[request.session_id] = future
    
#     mqtt_client_instance = app.state.mqtt_client
#     mqtt_client_instance.publish(
#         f"support/requests/{request.session_id}",
#         {
#             "question": request.question,
#             "session_id": request.session_id,
#             "timestamp": time.time()
#         }
#     )
    
#     try:
#         response = await asyncio.wait_for(future, timeout=30.0)
        
#         return ChatResponse(
#             question=request.question,
#             answer=response["answer"],
#             sources=[], 
#             session_id=request.session_id,
#             timestamp=response["timestamp"]
#         )
        
#     except asyncio.TimeoutError:
#         pending_requests.pop(request.session_id, None) 
#         raise HTTPException(status_code=504, detail="Request timed out.")
#     except Exception as e:
#         pending_requests.pop(request.session_id, None)
#         raise HTTPException(status_code=500, detail="Failed to process request")

# @app.post("/api/v1/chat/sync")
# def chat_sync(request: ChatRequest):
#     """Synchronous endpoint for easy testing"""
#     from agent import SupportAgent
#     agent = SupportAgent()
#     result = agent.run(request.question)
    
#     return ChatResponse(
#         question=request.question,
#         answer=result["answer"],
#         sources=[],
#         session_id=request.session_id,
#         timestamp=time.time()
#     )

# # Health Check Endpoint
# @app.get("/health")
# def health_check():
#     return {"status": "healthy"}















import uuid
import json
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from monitoring import USER_FEEDBACK
from schemas import ChatRequest, ChatResponse
from mqtt_client import MQTTClient

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-gateway")

# -------------------------------------------------
# App Lifespan
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API Gateway...")

    app.state.main_loop = asyncio.get_running_loop()
    app.state.pending_requests = {}

    mqtt_client = MQTTClient(broker_host="mosquitto")
    mqtt_client.connect()
    mqtt_client.client.loop_start()
    app.state.mqtt_client = mqtt_client

    listener_task = asyncio.create_task(mqtt_response_listener(app))

    try:
        yield
    finally:
        logger.info("Shutting down API Gateway...")
        listener_task.cancel()
        mqtt_client.client.loop_stop()
        mqtt_client.disconnect()

# -------------------------------------------------
# App Init
# -------------------------------------------------
app = FastAPI(title="Support API 2025", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""
    User feedback endpoint
"""
@app.post("/api/v1/feedback")
async def record_feedback(feedback: dict):
    # Now this works because it was imported!
    USER_FEEDBACK.labels(rating=feedback["rating"]).inc()
    return {"status": "recorded"}

# @app.get("/metrics")
# def metrics():
#     # This exposes ALL metrics from your entire app to Prometheus
#     return Response(generate_latest(), media_type="text/plain")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



# -------------------------------------------------
# MQTT Listener
# -------------------------------------------------
async def mqtt_response_listener(app: FastAPI):
    mqtt_client = app.state.mqtt_client

    def on_message(client, userdata, msg, properties=None):
        try:
            payload = json.loads(msg.payload.decode())
            
            # LOGIC FIX: If the message does not have an 'answer' or 'output', 
            # it is NOT a response from the worker. Ignore it!
            if "answer" not in payload and "output" not in payload:
                # Do NOT pop the future, just keep waiting
                return 

            print(f"GATEWAY RECEIVED ACTUAL RESPONSE: {payload}")
            session_id = payload.get("session_id")

            future = app.state.pending_requests.pop(session_id, None)
            if future and not future.done():
                app.state.main_loop.call_soon_threadsafe(
                    future.set_result, payload
                )
        except Exception as e:
            logger.exception("MQTT response handling failed")


    mqtt_client.client.on_message = on_message
    mqtt_client.client.subscribe("support/responses/+")
    # mqtt_client.subscribe("support/requests/#")
    logger.info("Subscribed to support/responses/+")

    while True:
        await asyncio.sleep(1)

# -------------------------------------------------
# Core Chat Handler
# -------------------------------------------------
async def handle_chat(request: ChatRequest) -> ChatResponse:
    if request.session_id == "default":
        request.session_id = f"http_{uuid.uuid4().hex[:12]}"

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
    except Exception:
        app.state.pending_requests.pop(request.session_id, None)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MQTT broker unreachable",
        )

    try:
        response = await asyncio.wait_for(future, timeout=600)
    except asyncio.TimeoutError:
        app.state.pending_requests.pop(request.session_id, None)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Agent did not respond in time",
        )

    return ChatResponse(
        question=request.question,
        answer=response.get("output") or response.get("answer") or "Error: Agent sent an empty response.",
        sources=response.get("sources", []),
        session_id=request.session_id,
        timestamp=response.get("timestamp", time.time()),
    )

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await handle_chat(request)

# ✅ Sync alias for frontend compatibility
@app.post("/api/v1/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    return await handle_chat(request)

@app.get("/health")
async def health():
    return {"status": "healthy", "time": time.time()}
