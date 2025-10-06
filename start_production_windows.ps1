# Script de inicio para producción - CLIP Comparador (Windows)
# Usar waitress en lugar de gunicorn para compatibilidad con Windows
# Uso: .\start_production_windows.ps1

Write-Host "Iniciando CLIP Comparador en modo PRODUCCION (Windows)..." -ForegroundColor Green

# Verificar que estamos en el directorio correcto
if (-not (Test-Path "app.py")) {
    Write-Host "Error: No se encuentra app.py. Ejecutar desde el directorio raiz del proyecto." -ForegroundColor Red
    exit 1
}

# Activar entorno virtual si existe
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    $pythonExe = "venv\Scripts\python.exe"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
    $pythonExe = ".venv\Scripts\python.exe"
} else {
    $pythonExe = "python"
}

# Configurar variables de entorno para producción
$env:FLASK_ENV = "production"
$env:FLASK_DEBUG = "False"
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"

# Verificar dependencias críticas
Write-Host "Verificando dependencias..." -ForegroundColor Yellow
& $pythonExe -c "import flask, torch, clip, PIL, waitress"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Faltan dependencias criticas. Ejecutar: pip install -r requirements.txt waitress" -ForegroundColor Red
    exit 1
}

# Verificar que el modelo CLIP esté disponible
Write-Host "Verificando modelo CLIP..." -ForegroundColor Yellow
$clipCode = @"
import clip
import torch
try:
    model, preprocess = clip.load('RN50', device='cpu')
    print('Modelo CLIP RN50 cargado correctamente')
except Exception as e:
    print(f'Error cargando modelo CLIP: {e}')
    exit(1)
"@

& $pythonExe -c $clipCode
if ($LASTEXITCODE -ne 0) {
    exit 1
}

# Verificar catálogo de embeddings
if (-not (Test-Path "catalogo\embeddings.json")) {
    Write-Host "Advertencia: No se encuentra catalogo\embeddings.json" -ForegroundColor Yellow
    Write-Host "Generando embeddings..." -ForegroundColor Yellow
    & $pythonExe generate_embeddings.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error generando embeddings" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Verificaciones completadas" -ForegroundColor Green

# Iniciar servidor Waitress
Write-Host "Iniciando servidor Waitress..." -ForegroundColor Green
Write-Host "URL: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Threads: 8" -ForegroundColor Cyan
Write-Host ""

# Crear script Python para iniciar waitress
$waitressScript = @"
import os
import sys
from waitress import serve
from app import app

if __name__ == '__main__':
    print('🌐 Servidor WSGI iniciado con Waitress')
    print('📍 Acceder a: http://localhost:5000')
    print('⏹️  Presionar Ctrl+C para detener')
    print('')
    
    # Configurar para producción
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    # Iniciar servidor
    serve(
        app,
        host='0.0.0.0',
        port=5000,
        threads=8,
        connection_limit=1000,
        cleanup_interval=30,
        channel_timeout=120
    )
"@

$waitressScript | Out-File -FilePath "production_server.py" -Encoding utf8

# Ejecutar servidor
& $pythonExe production_server.py