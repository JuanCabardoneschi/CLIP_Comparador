#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 PANEL DE ADMINISTRACIÓN GOODY - GESTIÓN DE METADATA
====================================================
Panel de control administrativo para gestionar:
- Categorías de productos
- Metadata de clasificaciones
- Estadísticas del sistema
- Configuración de parámetros

Autor: Sistema CLIP Comparador
Fecha: 03 Octubre 2025
Versión: 2.0.0
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_from_directory
import json
import os
from pathlib import Path
import shutil
from datetime import datetime
import statistics

app = Flask(__name__)
app.secret_key = 'admin_goody_2025_secure_key'

# 📂 Configuración de rutas
CATALOG_PATH = "catalogo"
CLASSIFICATIONS_FILE = "catalogo/product_classifications.json"
EMBEDDINGS_FILE = "catalogo/embeddings.json"
CONFIG_FILE = "admin_config.json"

# 🔐 Credenciales de administrador
ADMIN_CREDENTIALS = {
    'admin': 'clipadmin2025'
}

# 🖼️ Ruta estática para las imágenes del catálogo
@app.route('/static/catalogo/<filename>')
def catalog_images(filename):
    """Servir imágenes del catálogo"""
    return send_from_directory(CATALOG_PATH, filename)

# 🏷️ Configuración inicial de categorías
DEFAULT_CATEGORIES = [
    "DELANTAL",
    "AMBO VESTIR HOMBRE – DAMA",
    "CAMISAS HOMBRE- DAMA",
    "CASACAS",
    "ZUECOS",
    "GORROS – GORRAS",
    "CARDIGAN HOMBRE – DAMA",
    "BUZOS",
    "ZAPATO DAMA",
    "CHALECO DAMA- HOMBRE",
    "CHAQUETAS",
    "REMERAS"
]

# ==============================
# 🔐 FUNCIONES DE AUTENTICACIÓN
# ==============================

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Página de login administrativo"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
            session['admin_logged_in'] = True
            session['admin_username'] = username
            flash('✅ Acceso administrativo concedido', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('❌ Credenciales incorrectas', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    """Cerrar sesión administrativa"""
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    flash('👋 Sesión administrativa cerrada', 'info')
    return redirect(url_for('admin_login'))

def require_admin_login(f):
    """Decorador para requerir login administrativo"""
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('🔐 Acceso denegado. Requiere login administrativo.', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# ==============================
# 📊 FUNCIONES DE UTILIDAD
# ==============================

def load_config():
    """Cargar configuración administrativa"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Crear configuración por defecto
        default_config = {
            "categories": DEFAULT_CATEGORIES,
            "system_settings": {
                "auto_backup": True,
                "max_images": 1000,
                "confidence_threshold": 0.5
            },
            "metadata_fields": [
                {"name": "precio", "type": "number", "required": False},
                {"name": "descripcion", "type": "text", "required": False},
                {"name": "codigo_producto", "type": "text", "required": True},
                {"name": "stock", "type": "number", "required": False}
            ]
        }
        save_config(default_config)
        return default_config

def save_config(config):
    """Guardar configuración administrativa"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_classifications():
    """Cargar clasificaciones existentes"""
    try:
        with open(CLASSIFICATIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_classifications(classifications):
    """Guardar clasificaciones actualizadas"""
    with open(CLASSIFICATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(classifications, f, indent=2, ensure_ascii=False)

def get_system_stats():
    """Obtener estadísticas completas del sistema"""
    classifications = load_classifications()
    config = load_config()
    
    # Contar imágenes físicas
    catalog_dir = Path(CATALOG_PATH)
    total_images = len(list(catalog_dir.glob("*.jpg")))
    
    # Estadísticas de clasificación
    classified_count = len(classifications)
    unclassified_count = total_images - classified_count
    
    # Distribución por categorías
    category_distribution = {}
    confidence_scores = []
    
    for item in classifications.values():
        category = item.get('category', 'SIN CATEGORÍA')
        category_distribution[category] = category_distribution.get(category, 0) + 1
        confidence_scores.append(item.get('confidence', 0))
    
    # Estadísticas de confianza
    conf_stats = {}
    if confidence_scores:
        conf_stats = {
            'average': statistics.mean(confidence_scores),
            'median': statistics.median(confidence_scores),
            'min': min(confidence_scores),
            'max': max(confidence_scores)
        }
    
    return {
        'total_images': total_images,
        'classified': classified_count,
        'unclassified': unclassified_count,
        'classification_percentage': (classified_count / total_images * 100) if total_images > 0 else 0,
        'category_distribution': category_distribution,
        'confidence_stats': conf_stats,
        'total_categories': len(config['categories'])
    }

# ==============================
# 🏠 RUTAS PRINCIPALES
# ==============================

@app.route('/')
def index():
    """Redirigir al dashboard administrativo"""
    return redirect(url_for('admin_dashboard'))

@app.route('/admin')
@require_admin_login
def admin_dashboard():
    """Dashboard principal administrativo"""
    config = load_config()
    stats = get_system_stats()
    
    return render_template('admin_dashboard.html', 
                         config=config, 
                         stats=stats,
                         admin_user=session.get('admin_username'))

# ==============================
# 🏷️ GESTIÓN DE CATEGORÍAS
# ==============================

@app.route('/admin/categories')
@require_admin_login
def manage_categories():
    """Gestión de categorías de productos"""
    config = load_config()
    stats = get_system_stats()
    
    return render_template('admin_categories.html', 
                         categories=config['categories'],
                         distribution=stats['category_distribution'])

@app.route('/admin/api/categories', methods=['POST'])
@require_admin_login
def api_add_category():
    """API para agregar nueva categoría"""
    try:
        data = request.get_json()
        new_category = data.get('category', '').strip().upper()
        
        if not new_category:
            return jsonify({"success": False, "error": "Nombre de categoría requerido"})
        
        config = load_config()
        
        if new_category in config['categories']:
            return jsonify({"success": False, "error": "La categoría ya existe"})
        
        config['categories'].append(new_category)
        save_config(config)
        
        return jsonify({
            "success": True, 
            "message": f"Categoría '{new_category}' agregada exitosamente"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/admin/api/categories/<category>', methods=['DELETE'])
@require_admin_login
def api_delete_category(category):
    """API para eliminar categoría"""
    try:
        config = load_config()
        
        if category not in config['categories']:
            return jsonify({"success": False, "error": "Categoría no encontrada"})
        
        # Verificar si hay productos con esta categoría
        classifications = load_classifications()
        products_with_category = [k for k, v in classifications.items() 
                                if v.get('category') == category]
        
        if products_with_category:
            return jsonify({
                "success": False, 
                "error": f"No se puede eliminar. Hay {len(products_with_category)} productos con esta categoría"
            })
        
        config['categories'].remove(category)
        save_config(config)
        
        return jsonify({
            "success": True, 
            "message": f"Categoría '{category}' eliminada exitosamente"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ==============================
# 📋 GESTIÓN DE METADATA
# ==============================

@app.route('/admin/metadata')
@require_admin_login
def manage_metadata():
    """Gestión de metadata de productos"""
    classifications = load_classifications()
    config = load_config()
    
    # Preparar datos para la vista
    products_metadata = []
    categories_set = set()
    
    for filename, data in classifications.items():
        category = data.get('category', 'SIN CATEGORÍA')
        categories_set.add(category)
        
        product_info = {
            'filename': filename,
            'category': category,
            'confidence': data.get('confidence', 0),
            'metadata': data.get('metadata', {})
        }
        products_metadata.append(product_info)
    
    # Ordenar productos por nombre
    products_metadata.sort(key=lambda x: x['filename'])
    
    return render_template('admin_metadata.html', 
                         products=products_metadata,
                         metadata_fields=config['metadata_fields'],
                         categories=sorted(list(categories_set)))

@app.route('/admin/api/metadata/<filename>', methods=['POST'])
@require_admin_login
def api_update_metadata(filename):
    """API para actualizar metadata de un producto"""
    try:
        data = request.get_json()
        new_metadata = data.get('metadata', {})
        
        classifications = load_classifications()
        
        if filename not in classifications:
            return jsonify({"success": False, "error": "Producto no encontrado"})
        
        # Actualizar metadata
        if 'metadata' not in classifications[filename]:
            classifications[filename]['metadata'] = {}
        
        classifications[filename]['metadata'].update(new_metadata)
        save_classifications(classifications)
        
        return jsonify({
            "success": True, 
            "message": f"Metadata actualizada para {filename}"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ==============================
# ⚙️ CONFIGURACIÓN DEL SISTEMA
# ==============================

@app.route('/admin/settings')
@require_admin_login
def system_settings():
    """Configuración del sistema"""
    config = load_config()
    return render_template('admin_settings.html', config=config)

@app.route('/admin/api/settings', methods=['POST'])
@require_admin_login
def api_update_settings():
    """API para actualizar configuraciones del sistema"""
    try:
        data = request.get_json()
        config = load_config()
        
        # Actualizar configuraciones
        config['system_settings'].update(data.get('system_settings', {}))
        
        if 'metadata_fields' in data:
            config['metadata_fields'] = data['metadata_fields']
        
        save_config(config)
        
        return jsonify({
            "success": True, 
            "message": "Configuración actualizada exitosamente"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ==============================
# 🔄 BACKUP Y EXPORTACIÓN
# ==============================

@app.route('/admin/backup')
@require_admin_login
def backup_system():
    """Crear backup completo del sistema"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"backup_{timestamp}"
        
        # Crear directorio de backup
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copiar archivos importantes
        files_to_backup = [
            CLASSIFICATIONS_FILE,
            EMBEDDINGS_FILE,
            CONFIG_FILE
        ]
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                shutil.copy2(file_path, backup_dir)
        
        # Crear un subset del catálogo (primeras 10 imágenes como muestra)
        catalog_backup = os.path.join(backup_dir, "catalogo_sample")
        os.makedirs(catalog_backup, exist_ok=True)
        
        image_files = list(Path(CATALOG_PATH).glob("*.jpg"))[:10]
        for img_file in image_files:
            shutil.copy2(img_file, catalog_backup)
        
        return jsonify({
            "success": True, 
            "message": f"Backup creado exitosamente en: {backup_dir}",
            "backup_dir": backup_dir
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/admin/export/classifications')
@require_admin_login
def export_classifications():
    """Exportar clasificaciones como JSON"""
    classifications = load_classifications()
    
    from flask import Response
    response = Response(
        json.dumps(classifications, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-disposition": "attachment; filename=product_classifications_export.json"}
    )
    
    return response

@app.route('/admin/export/metadata')
@require_admin_login
def export_metadata():
    """Exportar metadata completa como JSON"""
    classifications = load_classifications()
    config = load_config()
    stats = get_system_stats()
    
    export_data = {
        "export_date": datetime.now().isoformat(),
        "system_stats": stats,
        "configuration": config,
        "classifications": classifications
    }
    
    from flask import Response
    response = Response(
        json.dumps(export_data, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-disposition": "attachment; filename=goody_metadata_complete.json"}
    )
    
    return response

# ==============================
# 🔍 API DE BÚSQUEDA
# ==============================

@app.route('/admin/api/search')
@require_admin_login
def api_search_products():
    """API para búsqueda de productos por diferentes criterios"""
    query = request.args.get('q', '').lower()
    category_filter = request.args.get('category', '')
    min_confidence = float(request.args.get('min_confidence', 0))
    
    classifications = load_classifications()
    results = []
    
    for filename, data in classifications.items():
        # Filtros de búsqueda
        if query and query not in filename.lower():
            continue
            
        if category_filter and data.get('category') != category_filter:
            continue
            
        if data.get('confidence', 0) < min_confidence:
            continue
        
        results.append({
            'filename': filename,
            'category': data.get('category'),
            'confidence': data.get('confidence'),
            'metadata': data.get('metadata', {})
        })
    
    return jsonify({"results": results, "total": len(results)})

if __name__ == '__main__':
    print("🔧 INICIANDO PANEL DE ADMINISTRACIÓN GOODY")
    print("=" * 60)
    print(f"📂 Ruta del catálogo: {CATALOG_PATH}")
    print(f"📄 Archivo de configuración: {CONFIG_FILE}")
    print(f"🔐 Usuario admin disponible")
    print("🌐 Panel disponible en: http://localhost:5001/admin")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)