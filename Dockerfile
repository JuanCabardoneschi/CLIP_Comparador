FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Azure
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install Python dependencies with Azure optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and static files
COPY . .

# Create necessary directories
RUN mkdir -p uploads templates static catalogo

# Set Azure-specific environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port for Azure Container Instances
EXPOSE 8000

# Health check for Azure
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/status || exit 1

# Start command optimized for Azure
CMD ["python", "app_simple.py"]