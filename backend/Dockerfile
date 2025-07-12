# Use Python 3.11 slim image for smaller size and security
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Cache buster: update this number to force rebuild
ARG CACHE_BUST=20250112-v4
RUN echo "Cache bust: $CACHE_BUST"
COPY requirements-minimal-test.txt requirements.txt
RUN echo "Installing Python dependencies..." && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    echo "Installing packages from requirements.txt..." && \
    pip install --no-cache-dir --timeout 300 --retries 3 -r requirements.txt && \
    echo "Dependencies installed successfully!"

# Copy application code
COPY . .

# Copy the DeFi dataset from local data directory
# (Note: data/defi_qa_dataset.json should be copied to backend/data/ before building)
COPY data /app/data

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create necessary directories
RUN mkdir -p cache logs persistent/cache persistent/logs persistent/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 