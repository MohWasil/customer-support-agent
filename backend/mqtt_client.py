import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import paho.mqtt.client as mqtt
import json
import asyncio
from prometheus_client import start_http_server
from typing import Callable, Dict
import uuid
import time

start_http_server(8001)
print("Worker metrics server started on port 8001")

class MQTTClient:
    # mosquitto
    def __init__(self, broker_host: str = "mosquitto", broker_port: int = 1883):
        self.client = mqtt.Client(client_id=f"agent_worker_{uuid.uuid4().hex[:8]}")
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.message_handlers: Dict[str, Callable] = {}
        # Create loop but don't run it yet
        self.loop = asyncio.new_event_loop()
        # Keep a reference to the thread where the loop will run
        self.loop_thread = None
        
        # Security: TLS config (for production)
        self.use_tls = False
        # get_rag_instance()
    def connect(self):
        """Connect to broker with error handling"""
        try:
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            # Connect
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            
            # Start network loop in background thread
            self.client.loop_start()
            print(f"MQTT Client connected to {self.broker_host}:{self.broker_port}")
            
        except Exception as e:
            print(f"MQTT Connection failed: {e}")
            raise
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful connection"""
        if rc == 0:
            print("MQTT connected successfully")
            # Subscribe to all request topics
            client.subscribe("support/requests/+")
        else:
            print(f"MQTT connection failed: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for disconnection"""
        print(f"MQTT disconnected: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Callback for incoming messages"""
        try:
            payload = json.loads(msg.payload.decode())
            topic = msg.topic
            
            # Extract user_id from topic: support/requests/{user_id}
            user_id = topic.split("/")[-1]
            
            print(f"Received message on {topic}: {payload}")
            
            # Route to handler
            if topic.startswith("support/requests/"):
                # Schedule the async handler on the dedicated loop
                # Ensure self.loop is running for this to work
                asyncio.run_coroutine_threadsafe(
                    self._handle_request(user_id, payload),
                    self.loop # Use the dedicated loop
                )
                
        except json.JSONDecodeError:
            print(f"Invalid JSON on topic {msg.topic}")
        except Exception as e:
            print(f"Message handling error: {e}")
    
    async def _handle_request(self, user_id: str, payload: dict):
        """Process request asynchronously"""
        try:
            from agent import SupportAgent
            agent = SupportAgent()
            
            question = payload.get("question")
            session_id = payload.get("session_id", user_id)
            
            # Generate response
            result = agent.run(question)
            
            # Publish response
            response_topic = f"support/responses/{user_id}"
            response_payload = {
                "session_id": session_id,
                "answer": result["answer"],
                "status": result["status"],
                "timestamp": time.time() 
            }
            
            # Publish using the sync method (runs in the MQTT client's thread context)
            # The _handle_request is async, but publish can be sync
            self.publish(response_topic, response_payload)
            
        except Exception as e:
            print(f"Agent processing error: {e}")
            import traceback
            traceback.print_exc() # Print full stack trace for debugging
            # Publish error response
            self.publish(
                f"support/responses/{user_id}",
                {
                    "session_id": payload.get("session_id", user_id),
                    "answer": "An error occurred processing your request.",
                    "status": "error",
                    "timestamp": time.time() # Now 'time' is imported
                }
            )
    
    def publish(self, topic: str, payload: dict):
        """Secure publish with JSON validation"""
        try:
            message = json.dumps(payload, ensure_ascii=False)
            # Use the MQTT client's publish method (it's thread-safe)
            # QoS 1 ensures at least once delivery attempt
            self.client.publish(topic, message, qos=1)
            print(f"Published to {topic}: {payload}") # Added payload for debugging
        except Exception as e:
            print(f"Publish error: {e}")
    
    def disconnect(self):
        """Graceful shutdown"""
        # Stop the MQTT loop first
        self.client.loop_stop()
        # Stop the asyncio loop if it's running
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop) 
        self.client.disconnect()
        print("MQTT client disconnected")

# Test MQTT Client
if __name__ == "__main__":
    mqtt_client = MQTTClient(broker_host="mosquitto")
    mqtt_client.connect()

    # Run the asyncio loop in the main thread (or a dedicated thread)
    # Option 1: Run the loop in the main thread
    print("Worker is now listening for requests...")
    print("Starting asyncio loop...")
    try:
        # This will block and run the asyncio event loop,
        # allowing scheduled tasks (like _handle_request) to run
        mqtt_client.loop.run_forever()
    except KeyboardInterrupt:
        print("Interrupted, stopping...")
    finally:
        mqtt_client.disconnect()
        mqtt_client.loop.close() # Close the loop after stopping
        print("Loop closed.")


    # Option 2 (Alternative): Run the loop in a background thread
    # import threading
    # def run_loop():
    #     asyncio.set_event_loop(mqtt_client.loop) # Set loop for this thread
    #     mqtt_client.loop.run_forever()
    # mqtt_client.loop_thread = threading.Thread(target=run_loop, daemon=True)
    # mqtt_client.loop_thread.start()
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     mqtt_client.disconnect()
    #     mqtt_client.loop.call_soon_threadsafe(mqtt_client.loop.stop) # Stop loop from main thread
    #     mqtt_client.loop_thread.join() # Wait for loop thread to finish
    #     mqtt_client.loop.close()
    #     print("Loop closed.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# import paho.mqtt.client as mqtt
# import json
# import asyncio
# from typing import Callable, Dict
# import uuid
# import time
# import logging

# # Import here but do NOT re-create per request
# from agent import SupportAgent

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("agent-mqtt")

# class MQTTClient:
#     def __init__(self, broker_host: str = "mosquitto", broker_port: int = 1883, agent_timeout: float = 30.0):
#         self.client = mqtt.Client(client_id=f"agent_worker_{uuid.uuid4().hex[:8]}")
#         self.broker_host = broker_host
#         self.broker_port = broker_port
#         self.message_handlers: Dict[str, Callable] = {}
#         self.loop = asyncio.new_event_loop()
#         self.loop_thread = None
#         self.use_tls = False

#         # Create the agent once (lazy: set to None, initialize in connect for better error handling)
#         self.agent: SupportAgent | None = None
#         self.agent_timeout = agent_timeout

#     def connect(self):
#         """Connect to broker with error handling and initialize the agent once."""
#         try:
#             # set callbacks
#             self.client.on_connect = self._on_connect
#             self.client.on_message = self._on_message
#             self.client.on_disconnect = self._on_disconnect

#             # Try to initialize the agent ONCE here so model loads during startup
#             try:
#                 logger.info("Initializing SupportAgent (this may take time if loading a model)...")
#                 self.agent = SupportAgent()
#                 logger.info("SupportAgent initialized successfully.")
#             except Exception as e:
#                 logger.exception("Failed to initialize SupportAgent at startup.")
#                 # keep going; _handle_request will handle missing agent

#             # Connect to MQTT Broker
#             self.client.connect(self.broker_host, self.broker_port, keepalive=60)
#             self.client.loop_start()
#             logger.info(f"MQTT Client connected to {self.broker_host}:{self.broker_port}")

#         except Exception as e:
#             logger.exception(f"MQTT Connection failed: {e}")
#             raise

#     def _on_connect(self, client, userdata, flags, rc):
#         if rc == 0:
#             logger.info("MQTT connected successfully")
#             client.subscribe("support/requests/+")
#         else:
#             logger.error(f"MQTT connection failed: {rc}")

#     def _on_disconnect(self, client, userdata, rc):
#         logger.warning(f"MQTT disconnected: {rc}")

#     def _on_message(self, client, userdata, msg):
#         try:
#             payload = json.loads(msg.payload.decode())
#             topic = msg.topic
#             user_id = topic.split("/")[-1]
#             logger.info(f"Received message on {topic}: {payload}")

#             if topic.startswith("support/requests/"):
#                 if not self.loop.is_running():
#                     # start loop in main thread or require it elsewhere; we assume main will run loop.run_forever()
#                     logger.info("Asyncio loop is not running; scheduling will still attempt to run.")
#                 asyncio.run_coroutine_threadsafe(
#                     self._handle_request(user_id, payload),
#                     self.loop
#                 )

#         except json.JSONDecodeError:
#             logger.error(f"Invalid JSON on topic {msg.topic}")
#         except Exception:
#             logger.exception("Message handling error")

#     async def _handle_request(self, user_id: str, payload: dict):
#         """Process request asynchronously; run blocking model code in thread with timeout."""
#         try:
#             # Ensure agent exists; create lazily if needed
#             if self.agent is None:
#                 try:
#                     logger.info("Lazy-initializing SupportAgent inside handler...")
#                     self.agent = SupportAgent()
#                     logger.info("SupportAgent lazy init successful.")
#                 except Exception:
#                     logger.exception("SupportAgent lazy initialization failed.")
#                     raise

#             question = payload.get("question")
#             session_id = payload.get("session_id", user_id)

#             # If agent.run is blocking (likely), run in thread so asyncio loop stays responsive
#             # Wrap in asyncio.wait_for to enforce timeout
#             try:
#                 result = await asyncio.wait_for(
#                     asyncio.to_thread(self.agent.run, question),
#                     timeout=self.agent_timeout
#                 )
#             except asyncio.TimeoutError:
#                 logger.error("Agent.run timed out.")
#                 raise RuntimeError("Model timeout")
#             except Exception:
#                 logger.exception("Agent.run raised an error")
#                 raise

#             # Normalize result
#             if isinstance(result, dict):
#                 answer = result.get("answer") or str(result)
#                 status = result.get("status", "success")
#             else:
#                 answer = str(result)
#                 status = "success"

#             response_topic = f"support/responses/{user_id}"
#             response_payload = {
#                 "session_id": session_id,
#                 "answer": answer,
#                 "status": status,
#                 "timestamp": time.time()
#             }

#             self.publish(response_topic, response_payload)

#         except Exception as e:
#             logger.exception(f"Agent processing error: {e}")
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
#         try:
#             message = json.dumps(payload, ensure_ascii=False)
#             self.client.publish(topic, message, qos=1)
#             logger.info(f"Published to {topic}: {payload}")
#         except Exception:
#             logger.exception("Publish error")

#     def disconnect(self):
#         self.client.loop_stop()
#         if self.loop and self.loop.is_running():
#             self.loop.call_soon_threadsafe(self.loop.stop)
#         self.client.disconnect()
#         logger.info("MQTT client disconnected")


# if __name__ == "__main__":
#     mqtt_client = MQTTClient()
#     mqtt_client.connect()
#     print("Starting asyncio loop...")
#     try:
#         # Run the loop in the main thread as before
#         mqtt_client.loop.run_forever()
#     except KeyboardInterrupt:
#         logger.info("Interrupted, stopping...")
#     finally:
#         mqtt_client.disconnect()
#         mqtt_client.loop.close()
#         logger.info("Loop closed.")
