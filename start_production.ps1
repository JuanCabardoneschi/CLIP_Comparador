# Script de inicio para producción - CLIP Comparador (Windows)
# Uso: .\start_production.ps1

Write-Host "Iniciando CLIP Comparador en modo PRODUCCION..." -ForegroundColor Green

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
    $gunicornExe = "venv\Scripts\gunicorn.exe"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
    $pythonExe = ".venv\Scripts\python.exe"
    $gunicornExe = ".venv\Scripts\gunicorn.exe"
} else {
    $pythonExe = "python"
    $gunicornExe = "gunicorn"
}

# Configurar variables de entorno para producción
$env:FLASK_ENV = "production"
$env:FLASK_DEBUG = "False"
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"

# Verificar dependencias críticas
Write-Host "Verificando dependencias..." -ForegroundColor Yellow
& $pythonExe -c "import flask, torch, clip, PIL"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Faltan dependencias criticas. Ejecutar: pip install -r requirements.txt" -ForegroundColor Red
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

# Calcular número de workers
$workers = [Math]::Min(4, (([Environment]::ProcessorCount * 2) + 1))

# Iniciar servidor Gunicorn
Write-Host "Iniciando servidor Gunicorn..." -ForegroundColor Green
Write-Host "URL: http://0.0.0.0:5000" -ForegroundColor Cyan
Write-Host "Workers: $workers" -ForegroundColor Cyan
Write-Host ""

# Ejecutar Gunicorn
& $gunicornExe --config gunicorn.conf.py app:app