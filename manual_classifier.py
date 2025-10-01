#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏷️ CLASIFICADOR MANUAL DE PRODUCTOS GOODY
===========================================
Interfaz web para clasificar manualmente cada imagen del catálogo.
Permite corregir clasificaciones incorrectas de forma visual e intuitiva.

Autor: Sistema CLIP Comparador
Fecha: 29 Septiembre 2025
Versión: 1.0.0
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import json
import os
from pathlib import Path

app = Flask(__name__)

# 🖼️ Ruta estática para las imágenes del catálogo
@app.route('/static/catalogo/<filename>')
def catalog_images(filename):
    """Servir imágenes del catálogo"""
    return send_from_directory(CATALOG_PATH, filename)

# 📂 Configuración de rutas
CATALOG_PATH = "catalogo"
CLASSIFICATIONS_FILE = "catalogo/product_classifications.json"

# 🏷️ Categorías disponibles
CATEGORIES = [
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

def get_all_images():
    """Obtener lista de todas las imágenes del catálogo"""
    images = []
    catalog_dir = Path(CATALOG_PATH)
    
    # Buscar todas las imágenes JPG en el catálogo
    for img_file in catalog_dir.glob("*.jpg"):
        images.append(img_file.name)
    
    return sorted(images)

@app.route('/')
def index():
    """Página principal - lista de imágenes para clasificar"""
    images = get_all_images()
    classifications = load_classifications()
    
    # Estadísticas
    total_images = len(images)
    classified_images = len([img for img in images if img in classifications])
    
    # Preparar datos para la vista
    image_data = []
    for img in images:
        current_category = "SIN CLASIFICAR"
        confidence = 0.0
        
        if img in classifications:
            current_category = classifications[img]["category"]
            confidence = classifications[img]["confidence"]
        
        image_data.append({
            "filename": img,
            "current_category": current_category,
            "confidence": confidence,
            "classified": img in classifications
        })
    
    return render_template('manual_classifier.html', 
                         image_data=image_data,
                         categories=CATEGORIES,
                         total_images=total_images,
                         classified_images=classified_images,
                         progress_percent=int((classified_images/total_images)*100) if total_images > 0 else 0)

@app.route('/classify/<filename>')
def classify_image(filename):
    """Página para clasificar una imagen específica"""
    if not filename.endswith('.jpg'):
        return redirect(url_for('index'))
    
    classifications = load_classifications()
    current_data = classifications.get(filename, {})
    
    return render_template('classify_single.html',
                         filename=filename,
                         categories=CATEGORIES,
                         current_category=current_data.get('category', ''),
                         current_confidence=current_data.get('confidence', 0.0),
                         top_3_categories=current_data.get('top_3_categories', []))

@app.route('/api/update_classification', methods=['POST'])
def update_classification():
    """API para actualizar la clasificación de una imagen"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        new_category = data.get('category')
        confidence = float(data.get('confidence', 1.0))
        
        if not filename or not new_category:
            return jsonify({"success": False, "error": "Datos incompletos"})
        
        # Cargar clasificaciones actuales
        classifications = load_classifications()
        
        # Actualizar o crear nueva clasificación
        if filename not in classifications:
            classifications[filename] = {
                "category": new_category,
                "confidence": confidence,
                "top_3_categories": [
                    {"category": new_category, "confidence": confidence}
                ]
            }
        else:
            # Actualizar clasificación existente
            old_category = classifications[filename]["category"]
            classifications[filename]["category"] = new_category
            classifications[filename]["confidence"] = confidence
            
            # Actualizar top_3_categories para que refleje el cambio manual
            classifications[filename]["top_3_categories"] = [
                {"category": new_category, "confidence": confidence},
                *[cat for cat in classifications[filename]["top_3_categories"] 
                  if cat["category"] != new_category][:2]
            ]
        
        # Guardar cambios
        save_classifications(classifications)
        
        return jsonify({
            "success": True, 
            "message": f"Clasificación actualizada: {filename} → {new_category}"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/statistics')
def get_statistics():
    """API para obtener estadísticas de clasificación"""
    images = get_all_images()
    classifications = load_classifications()
    
    stats = {
        "total_images": len(images),
        "classified_images": len([img for img in images if img in classifications]),
        "categories_count": {}
    }
    
    # Contar por categoría
    for img in images:
        if img in classifications:
            category = classifications[img]["category"]
            stats["categories_count"][category] = stats["categories_count"].get(category, 0) + 1
    
    return jsonify(stats)

@app.route('/api/delete_product', methods=['POST'])
def delete_product():
    """API para eliminar un producto (imagen + embedding + clasificación)"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({"success": False, "error": "Nombre de archivo requerido"})
        
        deleted_items = []
        
        # 1. Eliminar imagen física
        image_path = os.path.join(CATALOG_PATH, filename)
        if os.path.exists(image_path):
            os.remove(image_path)
            deleted_items.append("imagen")
        
        # 2. Eliminar de clasificaciones
        classifications = load_classifications()
        if filename in classifications:
            del classifications[filename]
            save_classifications(classifications)
            deleted_items.append("clasificación")
        
        # 3. Eliminar de embeddings
        embeddings_file = "catalogo/embeddings.json"
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'r') as f:
                    embeddings = json.load(f)
                
                # Buscar y eliminar el embedding
                embedding_path = os.path.join(CATALOG_PATH, filename)
                if embedding_path in embeddings:
                    del embeddings[embedding_path]
                    deleted_items.append("embedding")
                
                # Guardar embeddings actualizados
                with open(embeddings_file, 'w') as f:
                    json.dump(embeddings, f, indent=2)
                    
            except Exception as e:
                print(f"Error al actualizar embeddings: {e}")
        
        if deleted_items:
            return jsonify({
                "success": True, 
                "message": f"Producto eliminado: {filename}",
                "deleted_items": deleted_items
            })
        else:
            return jsonify({
                "success": False, 
                "error": "No se encontró el producto para eliminar"
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/export')
def export_classifications():
    """Exportar clasificaciones como JSON descargable"""
    classifications = load_classifications()
    
    from flask import Response
    import json
    
    response = Response(
        json.dumps(classifications, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-disposition": "attachment; filename=product_classifications_manual.json"}
    )
    
    return response

if __name__ == '__main__':
    print("🏷️ INICIANDO CLASIFICADOR MANUAL DE PRODUCTOS GOODY")
    print("=" * 60)
    print(f"📂 Ruta del catálogo: {CATALOG_PATH}")
    print(f"📄 Archivo de clasificaciones: {CLASSIFICATIONS_FILE}")
    print(f"🏷️ Categorías disponibles: {len(CATEGORIES)}")
    print("🌐 Servidor disponible en: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)