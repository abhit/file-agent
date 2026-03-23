FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install all heavy system dependencies required for vector databases (ChromaDB, hnswlib) 
# and document parsing (unstructured) on enterprise/custom architectures (like ARM or DGX)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    rustc \
    cargo \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .

# Use a longer timeout just in case it takes a long time to pull massive ML libraries 
# on the DGX server network
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the rest of the app's script
COPY app.py .

# Expose the standard Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
