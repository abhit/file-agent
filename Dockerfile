FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for vector databases (ChromaDB, hnswlib)
# and document parsing (unstructured) on ARM64/DGX architectures
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the app
COPY app.py .

# Create data directory so the volume mount has a target
RUN mkdir -p /app/data /app/chroma_db

# Expose the standard Streamlit port
EXPOSE 8501

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
