"""
CLIP Comparador - Aplicación Principal Modularizada
Sistema de búsqueda visual inteligente con modelo RN50 optimizado
"""

import os
import sys

# Importar configuración central
from config.settings import create_app, lazy_import_heavy_deps
from auth.authentication import setup_login_manager
from routes.main_routes import create_main_routes
from routes.auth_routes import create_auth_routes
from utils.helpers import initialize_system

def create_application():
    """Crear y configurar la aplicación Flask"""
    # Crear app con configuración base
    app = create_app()
    
    # Configurar autenticación
    login_manager = setup_login_manager(app)
    
    # Configurar rate limiting
    if lazy_import_heavy_deps():
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        
        limiter = Limiter(
            key_func=get_remote_address,
            app=app,
            default_limits=["200 per day", "50 per hour"]
        )
        
        # Rate limit específico para upload
        @limiter.limit("10 per minute")
        def upload_limit():
            pass
    
    # Registrar rutas
    create_main_routes(app)
    create_auth_routes(app)
    
    # Crear ruta adicional para admin (simplificada)
    @app.route('/admin/generate-embeddings', methods=['GET', 'POST'])
    def generate_embeddings():
        """Generar embeddings para nuevas imágenes del catálogo"""
        if lazy_import_heavy_deps():
            from flask import render_template, request, flash, redirect, url_for
            from flask_login import login_required
            
            if request.method == 'POST':
                # Aquí iría la lógica de generación de embeddings
                flash('Funcionalidad de generación de embeddings pendiente de implementar')
                return redirect(url_for('generate_embeddings'))
            
            return render_template('admin_generate.html')
        else:
            return "Error cargando dependencias", 500
    
    return app

# Crear la aplicación
app = create_application()

if __name__ == '__main__':
    if initialize_system():
        # Puerto dinámico para despliegue en cloud (Render, Heroku, etc.)
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        sys.exit(1)