"""
    Best client code.
"""

# import numpy as np
# if not hasattr(np, 'float_'):
#     np.float_ = np.float64

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import paho.mqtt.client as mqtt
# import json
# import asyncio
# from prometheus_client import start_http_server
# from typing import Callable, Dict
# import uuid
# import time

# start_http_server(8001)
# print("Worker metrics server started on port 8001")

# class MQTTClient:
#     # mosquitto
#     def __init__(self, broker_host: str = "mosquitto", broker_port: int = 1883):
#         self.client = mqtt.Client(client_id=f"agent_worker_{uuid.uuid4().hex[:8]}")
#         self.broker_host = broker_host
#         self.broker_port = broker_port
#         self.message_handlers: Dict[str, Callable] = {}
#         # Create loop
#         self.loop = asyncio.new_event_loop()
#         # Keep a reference to the thread where the loop will run
#         self.loop_thread = None
        
#         # Security: TLS config (for production)
#         self.use_tls = False
#         # get_rag_instance()
#     def connect(self):
#         """Connect to broker with error handling"""
#         try:
#             # Set callbacks
#             self.client.on_connect = self._on_connect
#             self.client.on_message = self._on_message
#             self.client.on_disconnect = self._on_disconnect
            
#             # Connect
#             self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            
#             # Start network loop in background thread
#             self.client.loop_start()
#             print(f"MQTT Client connected to {self.broker_host}:{self.broker_port}")
            
#         except Exception as e:
#             print(f"MQTT Connection failed: {e}")
#             raise
    
#     def _on_connect(self, client, userdata, flags, rc):
#         """Callback for successful connection"""
#         if rc == 0:
#             print("MQTT connected successfully")
#             # Subscribe to all request topics
#             client.subscribe("support/requests/+")
#         else:
#             print(f"MQTT connection failed: {rc}")
    
#     def _on_disconnect(self, client, userdata, rc):
#         """Callback for disconnection"""
#         print(f"MQTT disconnected: {rc}")
    
#     def _on_message(self, client, userdata, msg):
#         """Callback for incoming messages"""
#         try:
#             payload = json.loads(msg.payload.decode())
#             topic = msg.topic
            
#             # Extract user_id from topic:
#             user_id = topic.split("/")[-1]
            
#             print(f"Received message on {topic}: {payload}")
            
#             # Route to handler
#             if topic.startswith("support/requests/"):
#                 # Schedule the async handler on the dedicated loop
#                 asyncio.run_coroutine_threadsafe(
#                     self._handle_request(user_id, payload),
#                     self.loop # Use the dedicated loop
#                 )
                
#         except json.JSONDecodeError:
#             print(f"Invalid JSON on topic {msg.topic}")
#         except Exception as e:
#             print(f"Message handling error: {e}")
    
#     async def _handle_request(self, user_id: str, payload: dict):
#         """Process request asynchronously"""
#         try:
#             from agent import SupportAgent
#             agent = SupportAgent()
            
#             question = payload.get("question")
#             session_id = payload.get("session_id", user_id)
            
#             # Generate response
#             result = agent.run(question)
            
#             # Publish response
#             response_topic = f"support/responses/{user_id}"
#             response_payload = {
#                 "session_id": session_id,
#                 "answer": result["answer"],
#                 "status": result["status"],
#                 "timestamp": time.time() 
#             }
            
#             # Publish using the sync method 
#             self.publish(response_topic, response_payload)
            
#         except Exception as e:
#             print(f"Agent processing error: {e}")
#             import traceback
#             traceback.print_exc() 
#             # Publish error response
#             self.publish(
#                 f"support/responses/{user_id}",
#                 {
#                     "session_id": payload.get("session_id", user_id),
#                     "answer": "An error occurred processing your request.",
#                     "status": "error",
#                     "timestamp": time.time() 
#                 }
#             )
    
#     def publish(self, topic: str, payload: dict):
#         """Secure publish with JSON validation"""
#         try:
#             message = json.dumps(payload, ensure_ascii=False)
#             self.client.publish(topic, message, qos=1)
#             print(f"Published to {topic}: {payload}") 
#         except Exception as e:
#             print(f"Publish error: {e}")
    
#     def disconnect(self):
#         """Graceful shutdown"""
#         # Stop the MQTT loop first
#         self.client.loop_stop()
#         # Stop the asyncio loop if it's running
#         if self.loop and self.loop.is_running():
#             self.loop.call_soon_threadsafe(self.loop.stop) 
#         self.client.disconnect()
#         print("MQTT client disconnected")


# if __name__ == "__main__":
#     mqtt_client = MQTTClient(broker_host="mosquitto")
#     mqtt_client.connect()

#     print("Worker is now listening for requests...")
#     print("Starting asyncio loop...")
#     try:
#         # This will block and run the asyncio event loop,
#         mqtt_client.loop.run_forever()
#     except KeyboardInterrupt:
#         print("Interrupted, stopping...")
#     finally:
#         mqtt_client.disconnect()
#         mqtt_client.loop.close() # Close the loop after stopping
#         print("Loop closed.")









"""
    For production
"""
import numpy as np
import sys
import json
import asyncio
import uuid
import time
from typing import Callable, Dict
from loguru import logger
import paho.mqtt.client as mqtt
from prometheus_client import start_http_server

if not hasattr(np, 'float_'):
    np.float_ = np.float64

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # Fallback for environments where pysqlite3 isn't needed
    pass


# 2. Configure Production Logging
logger.remove()
logger.add(
    sys.stdout, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[session_id]}</cyan> - <level>{message}</level>",
    level="INFO"
)
# Default context for system-level logs
system_logger = logger.bind(session_id="SYSTEM-WORKER")

# Start Prometheus metrics
try:
    start_http_server(8001)
    system_logger.info("Worker metrics server started on port 8001")
except Exception as e:
    system_logger.error(f"Failed to start metrics server: {e}")

class MQTTClient:
    def __init__(self, broker_host: str = "mosquitto", broker_port: int = 1883):
        self.client = mqtt.Client(client_id=f"agent_worker_{uuid.uuid4().hex[:8]}")
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.loop = asyncio.new_event_loop()
        
    def connect(self):
        """Connect to broker with structured logging"""
        try:
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            self.message_handlers: Dict[str, Callable] = {}
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            system_logger.success(f"MQTT Client network loop started: {self.broker_host}")
        except Exception as e:
            system_logger.critical(f"MQTT Connection failed: {e}")
            raise

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            system_logger.success("Connected to MQTT Broker successfully.")
            client.subscribe("support/requests/+")
        else:
            system_logger.error(f"MQTT Connection Refused with code: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        system_logger.warning(f"MQTT disconnected from broker (Code: {rc})")

    def _on_message(self, client, userdata, msg):
        """Callback for incoming messages with session tracking"""
        try:
            payload = json.loads(msg.payload.decode())
            session_id = payload.get("session_id", "unknown")
            
            # Bind the session_id so every log from here on is traceable
            request_logger = logger.bind(session_id=session_id)
            request_logger.info(f"Received request on topic: {msg.topic}")

            if msg.topic.startswith("support/requests/"):
                asyncio.run_coroutine_threadsafe(
                    self._handle_request(session_id, payload),
                    self.loop
                )
        except json.JSONDecodeError:
            system_logger.error(f"Malformed JSON received on {msg.topic}")
        except Exception as e:
            system_logger.exception("Unexpected error in MQTT message callback")

    async def _handle_request(self, session_id: str, payload: dict):
        """Process agent logic with localized error handling"""
        request_logger = logger.bind(session_id=session_id)
        
        try:
            from agent import SupportAgent
            agent = SupportAgent()
            
            question = payload.get("question", "")
            request_logger.info(f"Invoking agent for query: {question[:30]}...")

            # Run the agent
            result = agent.run(question, session_id=session_id)
            
            response_payload = {
                "session_id": session_id,
                "answer": result["answer"],
                "status": result["status"],
                "timestamp": time.time() 
            }
            
            self.publish(f"support/responses/{session_id}", response_payload)
            request_logger.success("Agent response published back to Gateway.")
            
        except Exception as e:
            request_logger.error(f"Agent Logic Failure: {e}")
            self.publish(
                f"support/responses/{session_id}",
                {
                    "session_id": session_id,
                    "answer": "I encountered an internal error. Please try again later.",
                    "status": "error",
                    "timestamp": time.time() 
                }
            )

    def publish(self, topic: str, payload: dict):
        """Secure publish with session-aware logging"""
        session_id = payload.get("session_id", "SYSTEM")
        pub_logger = logger.bind(session_id=session_id)
        
        try:
            message = json.dumps(payload, ensure_ascii=False)
            self.client.publish(topic, message, qos=1)
        except Exception as e:
            pub_logger.error(f"Failed to publish message: {e}")

    def disconnect(self):
        system_logger.info("Gracefully disconnecting MQTT worker...")
        self.client.loop_stop()
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop) 
        self.client.disconnect()

if __name__ == "__main__":
    mqtt_client = MQTTClient(broker_host="mosquitto")
    mqtt_client.connect()

    system_logger.info("Worker is active. Listening for support requests...")
    try:
        mqtt_client.loop.run_forever()
    except KeyboardInterrupt:
        system_logger.warning("Worker stopped by user.")
    finally:
        mqtt_client.disconnect()
        mqtt_client.loop.close()
