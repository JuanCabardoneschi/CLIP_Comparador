#!/usr/bin/env python3
"""
CLIP Comparador - Servidor de producción para Railway
Usando Waitress WSGI server para eliminar warnings de development
"""

import os
import sys

# Configurar variables de entorno para Railway
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_DEBUG'] = 'False'

# Importar configuración central
from config.settings import create_app
from auth.authentication import setup_login_manager
from routes.main_routes import create_main_routes
from routes.auth_routes import create_auth_routes

def create_application():
    """Crear y configurar la aplicación Flask para Railway"""
    
    # Crear app con configuración base
    app = create_app()
    
    # Configuración específica para Railway
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    app.config['ENV'] = 'production'
    
    # Configurar puerto de Railway
    port = int(os.environ.get('PORT', 5000))
    app.config['PORT'] = port
    
    # Configurar autenticación
    login_manager = setup_login_manager(app)
    
    # Registrar rutas
    create_main_routes(app)
    create_auth_routes(app)
    
    print(f"🚀 CLIP Comparador configurado para Railway")
    print(f"🌐 Puerto: {port}")
    print(f"🔧 Modo: {app.config['ENV']}")
    
    return app

# Crear la aplicación
app = create_application()

if __name__ == '__main__':
    try:
        # Intentar usar waitress para producción
        from waitress import serve
        
        port = int(os.environ.get('PORT', 5000))
        host = '0.0.0.0'
        
        print(f"🌐 Iniciando servidor WSGI (Waitress) en {host}:{port}")
        print("✅ Servidor de PRODUCCIÓN - Sin warnings de development")
        
        # Iniciar servidor Waitress
        serve(
            app,
            host=host,
            port=port,
            threads=8,                # Número de hilos
            connection_limit=1000,    # Límite de conexiones
            cleanup_interval=30,      # Intervalo de limpieza
            channel_timeout=120,      # Timeout de canal
            url_scheme='http'
        )
        
    except ImportError:
        print("⚠️  Waitress no disponible, usando Flask development server")
        # Fallback a Flask development server
        port = int(os.environ.get('PORT', 5000))
        host = '0.0.0.0'
        
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )