# Dockerfile optimizado para Railway
FROM python:3.11-slim

# Configurar variables de entorno
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias Python
COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copiar el código de la aplicación
COPY . .

# Generar embeddings si no existen
RUN if [ ! -f "catalogo/embeddings.json" ]; then \
        echo "Generando embeddings..." && \
        python generate_embeddings.py; \
    fi

# Exponer puerto
EXPOSE $PORT

# Comando de inicio
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 --max-requests 1000 --preload app_railway:app