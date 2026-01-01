from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response, FastAPI



"""
    This code implements Observability, using Prometheus. It monitors the health, cost, and performance of the LLM application in real-time through Grafana dashboard. 
Counter (Requests/Tokens): Tracks total volume. It helps calculate Tokens per Minute (TPM) or Requests per Minute (RPM) to monitor API costs and usage spikes.
Histogram (Latency): Tracks how long the AI takes to respond. This is critical for identifying if the model provider is slowing down.
Gauge (Active Sessions): Tracks a value that goes up and down, showing how many users are currently interacting with the agent.
Labels: By using labels(model=model), you can compare different models side-by-side in your charts.
"""


# Metrics
AGENT_REQUESTS = Counter(
    'agent_requests_total',
    'Total requests to agent',
    ['model', 'status']
)

AGENT_LATENCY = Histogram(
    'agent_response_latency_seconds',
    'Response latency',
    ['model']
)

TOKEN_USAGE = Counter(
    'agent_tokens_total',
    'Total tokens used',
    ['model', 'type']  
)

USER_FEEDBACK = Counter(
    'user_feedback_total',
    'User feedback ratings',
    ['rating']  # thumbs_up / thumbs_down
)

ACTIVE_SESSIONS = Gauge('active_sessions', 'Number of active sessions')

def record_agent_metrics(model: str, latency: float, tokens_in: int, tokens_out: int, status: str):
    AGENT_LATENCY.labels(model=model).observe(latency)
    AGENT_REQUESTS.labels(model=model, status=status).inc()
    TOKEN_USAGE.labels(model=model, type='input').inc(tokens_in)
    TOKEN_USAGE.labels(model=model, type='output').inc(tokens_out)


# # Add metrics endpoint to api_mqtt.py
# app = FastAPI()
# @app.get("/metrics")
# def metrics():
#     return Response(generate_latest(), media_type="text/plain")