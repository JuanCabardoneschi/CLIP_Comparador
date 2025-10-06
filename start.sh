#!/bin/bash
# Script de inicio para Railway que maneja la variable PORT correctamente

# Usar PORT si está definida, sino usar 8000 por defecto
PORT=${PORT:-8000}

echo "Iniciando Gunicorn en puerto: $PORT"

# Ejecutar Gunicorn con el puerto correcto
exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 --max-requests 1000 --preload app_railway:app