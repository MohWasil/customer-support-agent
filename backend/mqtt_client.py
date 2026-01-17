"""
    MQTT Clinet with structured logging and error handling.
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
