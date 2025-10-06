#!/bin/bash
# Script de inicio para producción - CLIP Comparador

echo "🚀 Iniciando CLIP Comparador en modo PRODUCCIÓN..."

# Verificar que estamos en el directorio correcto
if [ ! -f "app.py" ]; then
    echo "❌ Error: No se encuentra app.py. Ejecutar desde el directorio raíz del proyecto."
    exit 1
fi

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "📦 Activando entorno virtual..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "📦 Activando entorno virtual..."
    source .venv/bin/activate
fi

# Configurar variables de entorno para producción
export FLASK_ENV=production
export FLASK_DEBUG=False
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verificar dependencias críticas
echo "🔍 Verificando dependencias..."
python -c "import flask, torch, clip, PIL" || {
    echo "❌ Error: Faltan dependencias críticas. Ejecutar: pip install -r requirements.txt"
    exit 1
}

# Verificar que el modelo CLIP esté disponible
echo "🧠 Verificando modelo CLIP..."
python -c "
import clip
import torch
try:
    model, preprocess = clip.load('RN50', device='cpu')
    print('✅ Modelo CLIP RN50 cargado correctamente')
except Exception as e:
    print(f'❌ Error cargando modelo CLIP: {e}')
    exit(1)
" || exit 1

# Verificar catálogo de embeddings
if [ ! -f "catalogo/embeddings.json" ]; then
    echo "⚠️  Advertencia: No se encuentra catalogo/embeddings.json"
    echo "   Generando embeddings..."
    python generate_embeddings.py || {
        echo "❌ Error generando embeddings"
        exit 1
    }
fi

echo "✅ Verificaciones completadas"

# Iniciar servidor Gunicorn
echo "🌐 Iniciando servidor Gunicorn..."
echo "📍 URL: http://0.0.0.0:5000"
echo "🔄 Workers: $(python -c "import multiprocessing; print(min(4, (multiprocessing.cpu_count() * 2) + 1))")"
echo ""

exec gunicorn --config gunicorn.conf.py app:app