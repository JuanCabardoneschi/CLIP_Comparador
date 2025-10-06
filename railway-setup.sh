#!/bin/bash
# Script de configuración para Railway deployment
# Este script se ejecuta automáticamente en el build de Railway

echo "🚀 Iniciando configuración para Railway..."

# Configurar variables de entorno
export FLASK_ENV=production
export FLASK_DEBUG=False
export PYTHONPATH=$PYTHONPATH:/app

# Verificar dependencias críticas
echo "🔍 Verificando dependencias..."
python -c "import flask, torch, clip, PIL; print('✅ Dependencias verificadas')"

# Generar embeddings si no existen
if [ ! -f "catalogo/embeddings.json" ]; then
    echo "📊 Generando embeddings del catálogo..."
    python generate_embeddings.py
    if [ $? -eq 0 ]; then
        echo "✅ Embeddings generados correctamente"
    else
        echo "❌ Error generando embeddings"
        exit 1
    fi
else
    echo "✅ Embeddings ya existen"
fi

echo "✅ Configuración completada para Railway"