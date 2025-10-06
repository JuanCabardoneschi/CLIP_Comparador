# 🚀 CLIP Comparador - Guía de Producción

## Configuración para Producción

### 📋 Prerrequisitos

1. **Python 3.8+** instalado
2. **Dependencias instaladas:** `pip install -r requirements.txt`
3. **Catálogo de imágenes** en la carpeta `catalogo/`
4. **Embeddings generados:** `python generate_embeddings.py`

### 🏭 Servidor de Producción (Gunicorn)

#### Windows (PowerShell)
```powershell
# Dar permisos de ejecución al script
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Iniciar en modo producción
.\start_production.ps1
```

#### Linux/macOS (Bash)
```bash
# Dar permisos de ejecución
chmod +x start_production.sh

# Iniciar en modo producción
./start_production.sh
```

#### Manual (cualquier sistema)
```bash
# Configurar variables de entorno
export FLASK_ENV=production
export FLASK_DEBUG=False

# Iniciar con Gunicorn
gunicorn --config gunicorn.conf.py app:app
```

### ⚙️ Configuración de Gunicorn

El archivo `gunicorn.conf.py` incluye:

- **Workers:** Calculados automáticamente según CPU disponible
- **Timeout:** 120 segundos (suficiente para carga de modelos CLIP)
- **Memoria compartida:** Optimizada para modelos ML
- **Logging:** Estructurado para producción
- **Seguridad:** Límites de request configurados

### 🔧 Variables de Entorno

| Variable | Desarrollo | Producción | Descripción |
|----------|------------|------------|-------------|
| `FLASK_ENV` | `development` | `production` | Modo de Flask |
| `FLASK_DEBUG` | `True` | `False` | Debugging |
| `SECRET_KEY` | (auto) | **⚠️ CONFIGURAR** | Clave secreta |

### 🌐 URLs y Puertos

- **Desarrollo:** `http://127.0.0.1:5000`
- **Producción:** `http://0.0.0.0:5000` (accesible desde red)

### 📊 Monitoreo

#### Verificar estado del servidor
```bash
curl http://localhost:5000/status
```

#### Logs en tiempo real
```bash
tail -f /var/log/gunicorn/access.log
```

### 🔒 Seguridad en Producción

1. **SECRET_KEY:** Configurar variable de entorno única
2. **HTTPS:** Configurar certificados SSL/TLS
3. **Firewall:** Restringir acceso a puertos necesarios
4. **Reverse Proxy:** Usar Nginx/Apache como proxy
5. **Rate Limiting:** Incluido en la aplicación

### 🚨 Resolución de Problemas

#### Error: "No se encuentra app.py"
```bash
# Ejecutar desde el directorio raíz del proyecto
cd /ruta/a/CLIP_Comparador
./start_production.sh
```

#### Error: "Faltan dependencias"
```bash
pip install -r requirements.txt
```

#### Error: "No se encuentra modelo CLIP"
```bash
# El modelo se descarga automáticamente la primera vez
python -c "import clip; clip.load('RN50')"
```

#### Error: "No hay embeddings"
```bash
python generate_embeddings.py
```

### 📈 Optimización

1. **CPU:** Gunicorn ajusta workers automáticamente
2. **Memoria:** Preload activado para compartir modelo CLIP
3. **Disco:** Embeddings en memoria para acceso rápido
4. **Red:** Keep-alive configurado

### 🔄 Reinicio y Actualizaciones

```bash
# Reinicio suave (sin downtime)
kill -HUP $(pgrep -f "gunicorn.*app:app")

# Reinicio completo
pkill -f "gunicorn.*app:app"
./start_production.sh
```

## 🎯 Diferencias vs Desarrollo

| Aspecto | Desarrollo | Producción |
|---------|------------|------------|
| Servidor | Flask dev server | Gunicorn WSGI |
| Workers | 1 thread | Múltiples workers |
| Debug | Activado | Desactivado |
| Reload | Automático | Manual |
| Logs | Consola | Estructurados |
| Seguridad | Básica | Hardened |
| Performance | Básica | Optimizada |

## ✅ Checklist de Producción

- [ ] Variables de entorno configuradas
- [ ] SECRET_KEY único generado
- [ ] Embeddings del catálogo generados
- [ ] Gunicorn instalado y funcionando
- [ ] Logs configurados
- [ ] Monitoring configurado
- [ ] Backups programados
- [ ] Certificados SSL (si aplicable)
- [ ] Reverse proxy configurado (si aplicable)
- [ ] Firewall configurado