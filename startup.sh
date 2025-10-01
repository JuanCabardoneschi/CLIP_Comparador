#!/bin/bash
# Startup script para Azure Web Apps

echo "🔧 Configurando CLIP Comparador para Azure..."

# Crear directorio de uploads si no existe
mkdir -p uploads

# Verificar embeddings
if [ -f "catalogo/embeddings.json" ]; then
    echo "✅ Embeddings encontrados"
else
    echo "⚠️  Generando embeddings del catálogo..."
    python generate_embeddings.py
fi

echo "🚀 Iniciando aplicación..."
python app_simple.py