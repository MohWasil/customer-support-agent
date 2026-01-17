import os
import uvicorn

# Read the ROLE variable set in docker-compose.yml
role = os.getenv("ROLE", "gateway") 

if role == "worker":
    # IMPORT the logic for your Agent/MQTT subscriber
    from agent import SupportAgent 
    SupportAgent() 
else:
    from api_mqtt import app 
    uvicorn.run(app, host="0.0.0.0", port=8000)
