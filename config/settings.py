"""
Configuración central del sistema CLIP Comparador
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# 🏷️ Sistema de Versioning Automático
VERSION = "3.9.6"
BUILD_DATE = "2025-10-05"
CHANGES_LOG = {
    "3.9.7": "PRODUCCIÓN: Configuración Gunicorn + scripts de inicio para servidor WSGI de producción",
    "3.9.6": "FIX EXIF: Corregir import error + detectión automática por dimensiones como fallback para móviles",
    "3.9.5": "DEBUG: Logs detallados para diagnosticar problema de orientación EXIF en móviles",
    "3.9.4": "FIX EXIF: Evitar doble corrección orientación - solo aplicar en archivos, no en objetos Image ya procesados",
    "3.9.3": "NUEVA FUNCIONALIDAD: Corrección automática de orientación EXIF para imágenes de móviles (rotación 90°)",
    "3.9.2": "FIX RUTAS IMÁGENES: Normalizar separadores \\ a / antes de basename() para compatibilidad Linux/Windows",
    "3.9.1": "FIX COMPLETO JSON: Convertir float32 en calculate_similarity y results para evitar errores serialización",
    "3.9.0": "FIX JSON SERIALIZATION: Convertir float32 PyTorch a float Python para evitar error 'not JSON serializable'",
    "3.8.9": "FIX CRÍTICO CATEGORÍAS: Corregido bucle classifications + generadas product_classifications.json para detección de productos",
    "3.8.8": "FIX DETECCIÓN CATEGORÍAS: Mejorada lógica para detectar 'camisa' en 'camisa con botones y cuello'",
    "3.8.7": "FIX COMPATIBILIDAD: Removido half precision problemático + estado de modelo corregido",
    "3.8.6": "CORRECCIÓN CRÍTICA: RN50 (244MB) en lugar de ViT-B/32 (338MB) - Error de tamaños de modelos",
    "3.8.5": "OPTIMIZACIÓN MEMORIA: Sistema optimizado para 512MB RAM con lazy loading y garbage collection",
    "3.8.0": "DETECCIÓN AMPLIADA: Agregadas categorías no comercializadas para correcta identificación",
    "3.7.0": "ENFOQUE SIMPLIFICADO: Verificación genérica de categorías comercializadas vs no comercializadas"
}

# Variables globales para lazy loading
model = None
preprocess = None
device = None
catalog_embeddings = {}

def lazy_import_heavy_deps():
    """Importar dependencias pesadas solo cuando sea necesario"""
    global torch, clip, np, Image, Flask, render_template, request, jsonify
    global send_from_directory, redirect, url_for, flash, session
    global LoginManager, UserMixin, login_user, login_required, logout_user, current_user
    global Limiter, get_remote_address
    
    try:
        # Importaciones básicas de Flask (siempre deben estar disponibles)
        from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
        
        # Importaciones de ML (pueden fallar si no están instaladas)
        import torch
        import clip  
        import numpy as np
        from PIL import Image
        
        # Importaciones de autenticación
        from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
        
        # Rate limiting (opcional)
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address
        except ImportError:
            # Rate limiting es opcional
            Limiter = None
            get_remote_address = None
        
        return True
    except ImportError as e:
        print(f"Error importando dependencias: {e}")
        return False

def show_version_info():
    """Mostrar información de versión"""
    pass

def create_app():
    """Crear aplicación Flask con lazy loading"""
    # Solo importar Flask básico
    try:
        from flask import Flask
    except ImportError:
        raise Exception("Flask no está disponible")
    
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    
    # 🏭 CONFIGURACIÓN DE PRODUCCIÓN
    is_production = os.environ.get('FLASK_ENV') == 'production'
    
    if is_production:
        # Configuración para producción
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
        app.config['PROPAGATE_EXCEPTIONS'] = True
        app.config['SESSION_COOKIE_SECURE'] = True  # Solo HTTPS en producción
        app.config['SESSION_COOKIE_HTTPONLY'] = True
        app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
        print("🏭 Modo PRODUCCIÓN activado")
    else:
        # Configuración para desarrollo
        app.config['DEBUG'] = True
        app.config['TESTING'] = False
        print("🔧 Modo DESARROLLO activado")
    
    # Configuraciones comunes
    app.config['CATALOGO_FOLDER'] = 'catalogo'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clip-demo-secret-key-2025-very-secure')
    
    print(f"Flask app created with SECRET_KEY: {app.config['SECRET_KEY'][:10]}...")
    
    return app

# 🚀 CONFIGURACIÓN PARA GUNICORN
def get_wsgi_app():
    """
    Función de entrada para Gunicorn
    Usado en: gunicorn --config gunicorn.conf.py app:app
    """
    return create_app()