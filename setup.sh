#!/bin/bash
# Script de configuración para Render

echo "🔧 Configurando CLIP Comparador para producción..."

# Crear directorio de uploads si no existe
mkdir -p uploads

# Verificar que embeddings.json existe
if [ -f "catalogo/embeddings.json" ]; then
    echo "✅ Embeddings encontrados"
else
    echo "⚠️  Generando embeddings del catálogo..."
    python generate_embeddings.py
fi

echo "🚀 Configuración completada"