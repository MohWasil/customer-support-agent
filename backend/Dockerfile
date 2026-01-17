FROM python:3.10.9-slim


RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libmagic-dev
RUN pip install "pydantic>=2.9.0" "typing_extensions>=4.12.0"
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu 
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dockerfile snippet
WORKDIR /app
COPY ./data/knowledge_base /app/data/knowledge_base

# Copy application code
COPY . .


RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api_mqtt:app", "--host", "0.0.0.0", "--port", "8000"]