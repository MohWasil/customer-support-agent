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
        # Create loop
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
            
            # Extract user_id from topic:
            user_id = topic.split("/")[-1]
            
            print(f"Received message on {topic}: {payload}")
            
            # Route to handler
            if topic.startswith("support/requests/"):
                # Schedule the async handler on the dedicated loop
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
            
            # Publish using the sync method 
            self.publish(response_topic, response_payload)
            
        except Exception as e:
            print(f"Agent processing error: {e}")
            import traceback
            traceback.print_exc() 
            # Publish error response
            self.publish(
                f"support/responses/{user_id}",
                {
                    "session_id": payload.get("session_id", user_id),
                    "answer": "An error occurred processing your request.",
                    "status": "error",
                    "timestamp": time.time() 
                }
            )
    
    def publish(self, topic: str, payload: dict):
        """Secure publish with JSON validation"""
        try:
            message = json.dumps(payload, ensure_ascii=False)
            self.client.publish(topic, message, qos=1)
            print(f"Published to {topic}: {payload}") 
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

    print("Worker is now listening for requests...")
    print("Starting asyncio loop...")
    try:
        # This will block and run the asyncio event loop,
        mqtt_client.loop.run_forever()
    except KeyboardInterrupt:
        print("Interrupted, stopping...")
    finally:
        mqtt_client.disconnect()
        mqtt_client.loop.close() # Close the loop after stopping
        print("Loop closed.")
