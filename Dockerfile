FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and documents
COPY . .

# Generate ChromaDB vector database during Docker build (if documents/ present)
RUN python ingest.py

# Expose default port
EXPOSE 8001

# Run FastAPI server listening on 0.0.0.0 and PORT environment variable (Render / Railway compatible)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
