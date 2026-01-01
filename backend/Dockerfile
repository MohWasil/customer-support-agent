# FROM python:3.11-slim

# # Security: Create non-root user
# RUN useradd -m -u 1000 appuser

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Create non-root user ownership
# RUN chown -R appuser:appuser /app
# USER appuser

# # Health check
# HEALTHCHECK --interval=30s --timeout=3s \
#     CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# EXPOSE 8000

# CMD ["uvicorn", "api_mqtt:app", "--host", "0.0.0.0", "--port", "8000"]






FROM python:3.10.9-slim

# Security: Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the parent directory (project root)
# The path is relative to the build context (./backend when run via docker-compose build)
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# RUN pip install --no-cache-dir Jinja2 typing-extensions
RUN apt-get update && apt-get install -y libmagic-dev
RUN pip install "pydantic>=2.9.0" "typing_extensions>=4.12.0"
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu 
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dockerfile snippet
WORKDIR /app
COPY ./data/knowledge_base /app/data/knowledge_base
# Pre-calculate the vector DB during build to save startup time
# RUN python -c "from rag_with_memory import MemoryRAG; MemoryRAG('./app/data/knowledge_base')"

# Copy application code (from the current build context directory ./backend)
COPY . .

# Create non-root user ownership
RUN chown -R appuser:appuser /app
USER appuser

# Health check
# Note: This assumes a /health endpoint exists in your FastAPI app
# If not, you might need to add one or adjust the health check command
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

# CMD ["uvicorn", "api_mqtt:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "-m", "uvicorn", "api_mqtt:app", "--host", "0.0.0.0", "--port", "8000"]