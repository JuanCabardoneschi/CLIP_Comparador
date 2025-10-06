"""
Rutas principales de la aplicación
"""

import os
from config.settings import lazy_import_heavy_deps
from core.image_processing import get_image_embedding
from core.classification import classify_query_image
from core.search_engine import find_similar_images

def create_main_routes(app):
    """Crear las rutas principales"""
    if not lazy_import_heavy_deps():
        raise Exception("No se pudieron cargar las dependencias")
    
    from flask import render_template, request, jsonify
    from flask_login import login_required
    from PIL import Image
    import json
    
    @app.route('/')
    @login_required
    def index():
        from flask_login import current_user
        return render_template('index.html', user=current_user.id if current_user.is_authenticated else None)
    
    @app.route('/upload', methods=['POST'])
    @login_required  
    def upload_file():
        try:
            print("=== INICIO UPLOAD ===")
            if 'file' not in request.files:
                print("Error: No se recibió archivo")
                return jsonify({'error': 'No se recibió archivo'}), 400
            
            file = request.files['file']
            if file.filename == '':
                print("Error: No se seleccionó archivo")
                return jsonify({'error': 'No se seleccionó archivo'}), 400
            
            print(f"Archivo recibido: {file.filename}, tipo: {file.content_type}")
            
            # Leer imagen desde memoria
            print("Cargando imagen...")
            image = Image.open(file.stream)
            print(f"Imagen cargada: {image.size}, modo: {image.mode}")
            
            # Convertir imagen a base64 para mostrar en frontend
            import base64
            import io
            img_buffer = io.BytesIO()
            # Asegurar que la imagen esté en RGB para JPEG
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            uploaded_image_data = f"data:image/jpeg;base64,{img_base64}"
            
            # Procesar imagen
            print("Generando embedding...")
            query_embedding = get_image_embedding(image)
            
            if query_embedding is None:
                print("Error: get_image_embedding devolvió None")
                return jsonify({'error': 'Error procesando imagen'}), 500
            
            print(f"Embedding generado: shape {query_embedding.shape}")
            
            # Clasificar automáticamente
            print("Clasificando imagen...")
            query_type, query_confidence = classify_query_image(image)
            print(f"Clasificación: {query_type}, confianza: {query_confidence}")
            
            # Buscar imágenes similares
            print("Buscando imágenes similares...")
            results = find_similar_images(query_embedding, 3, query_type, query_confidence)
            
            # Verificar si es una categoría no comercializada
            if results == "CATEGORIA_NO_COMERCIALIZADA":
                return jsonify({
                    'similar_images': [],
                    'query_type': query_type,
                    'query_confidence': float(query_confidence) if query_confidence else 0.0,
                    'total_results': 0,
                    'friendly_message': f'Hemos detectado que buscas "{query_type}". Actualmente nuestro catálogo se especializa en uniformes profesionales, delantales, camisas de trabajo, chaquetas, buzos, gorros y calzado laboral. ¡Te invitamos a explorar nuestros productos disponibles!',
                    'category_not_available': True,
                    'uploaded_image_data': uploaded_image_data  # Incluir imagen también aquí
                }), 200
            
            print(f"Resultados encontrados: {len(results)}")
            
            # Verificar si no hay resultados por categoría no comercializada (código legacy - ya no debería ejecutarse)
            if len(results) == 0 and query_type:
                # Importar configuración centralizada
                from config.categories import get_commercial_categories, get_non_commercial_categories
                
                # Obtener categorías desde configuración centralizada
                categorias_goody = get_commercial_categories()
                categorias_no_comercializadas = get_non_commercial_categories()
                
                # Verificar si es una categoría explícitamente no comercializada
                query_lower = query_type.lower()
                for categoria_no in categorias_no_comercializadas:
                    if categoria_no in query_lower:
                        return jsonify({
                            'similar_images': [],
                            'query_type': query_type,
                            'query_confidence': float(query_confidence) if query_confidence else 0.0,
                            'total_results': 0,
                            'friendly_message': f'Hemos detectado que buscas "{query_type}". Actualmente nuestro catálogo se especializa en uniformes profesionales, delantales, camisas de trabajo, chaquetas, buzos, gorros y calzado laboral. ¡Te invitamos a explorar nuestros productos disponibles!',
                            'category_not_available': True
                        }), 200  # Cambiar a 200 (éxito) en lugar de 400 (error)
                
                categoria_encontrada = any(categoria in query_lower for categoria in categorias_goody)
                
                if not categoria_encontrada:
                    return jsonify({
                        'similar_images': [],
                        'query_type': query_type,
                        'query_confidence': float(query_confidence) if query_confidence else 0.0,
                        'total_results': 0,
                        'friendly_message': f'Hemos detectado que buscas "{query_type}". Actualmente nuestro catálogo se especializa en uniformes profesionales, delantales, camisas de trabajo, chaquetas, buzos, gorros y calzado laboral. ¡Te invitamos a explorar nuestros productos disponibles!',
                        'category_not_available': True
                    }), 200  # Cambiar a 200 (éxito) en lugar de 400 (error)
            
            # Formatear resultados para el frontend
            formatted_results = []
            for filename, score in results:
                formatted_results.append({
                    'filename': filename.replace('catalogo\\', '').replace('catalogo/', ''),  # Limpiar path
                    'similarity': float(score),
                    'similarity_percent': f"{score*100:.1f}"
                })
            
            response_data = {
                'similar_images': formatted_results,  # Campo que espera el frontend
                'query_type': query_type,  # Campo que espera el frontend 
                'query_confidence': float(query_confidence) if query_confidence else 0.0,
                'total_results': len(formatted_results),
                'uploaded_image_data': uploaded_image_data  # Imagen en base64 para mostrar
            }
            
            print("=== UPLOAD EXITOSO ===")
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/status')
    def status():
        """Endpoint de estado del sistema"""
        from models.clip_model import get_model, get_device
        from core.classification import get_catalog_embeddings
        from config.settings import VERSION
        from flask_login import current_user
        import datetime
        
        model = get_model()
        catalog_embeddings = get_catalog_embeddings()
        device = get_device()
        
        status_data = {
            'version': VERSION,
            'build_date': 'undefined',
            'latest_changes': 'Sistema modularizado con Flask-Login',
            'model_loaded': model is not None,
            'catalog_size': len(catalog_embeddings),  # Campo que espera el JavaScript
            'embeddings_count': len(catalog_embeddings),  # Campo adicional
            'device': device or 'cpu',
            'user_logged_in': current_user.is_authenticated if current_user else False,
            'username': current_user.id if current_user and current_user.is_authenticated else None,
            'status': 'OK' if model is not None else 'Model not loaded'
        }
        return jsonify(status_data)
    
    @app.route('/catalogo/<filename>')
    def serve_catalog_image(filename):
        """Servir imágenes del catálogo"""
        from flask import send_from_directory
        import os
        
        # Usar ruta absoluta para el catálogo
        catalog_path = os.path.join(os.getcwd(), 'catalogo')
        print(f"Intentando servir: {filename} desde {catalog_path}")
        
        return send_from_directory(catalog_path, filename)