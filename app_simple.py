"""
CLIP Comparador - Versión 3.8.1 - OPTIMIZADO PARA 512MB RAM
Sistema de búsqueda visual inteligente con modelo RN50 optimizado
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

# Variables globales para lazy loading
model = None
preprocess = None
device = None


def lazy_import_heavy_deps():
    """Importar dependencias pesadas solo cuando sea necesario"""
    global torch, clip, np, Image, Flask, render_template, request, jsonify
    global send_from_directory, redirect, url_for, flash, session
    global LoginManager, UserMixin, login_user, login_required
    global logout_user, current_user, Limiter, get_remote_address

    try:
        import torch
        import clip
        import numpy as np
        from PIL import Image
        from flask import (Flask, render_template, request, jsonify,
                           send_from_directory, redirect, url_for, flash,
                           session)
        from flask_login import (LoginManager, UserMixin, login_user,
                                 login_required, logout_user, current_user)
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address

        return True
    except ImportError:
        return False


# 🏷️ Sistema de Versioning Automático
VERSION = "3.9.17"
BUILD_DATE = "2025-10-06"
CHANGES_LOG = {
    "3.9.17": ("FIX LOGIN RATE LIMIT: Aumentado límite de login de 5/min a 20/min. "
               "Elimina el bloqueo 'Too Many Requests' en la página de login"),
    "3.9.16": ("CÓDIGO SIMPLIFICADO: Eliminada lógica compleja innecesaria de recarga "
               "automática, contador de consultas y limpieza agresiva. Mantenidos solo "
               "los límites de rate limiting corregidos que solucionan el problema real. "
               "Código más limpio y mantenible"),
    "3.9.15": ("FIX RATE LIMITING: Aumentados límites de consultas de 10/min a 50/min "
               "globalmente y de 20/min a 100/min en upload. Soluciona el bloqueo "
               "después de varios usos consecutivos"),
    "3.9.14": ("FIX MEMORIA PERSISTENTE: Implementado sistema de recarga automática "
               "del modelo cada 15 consultas, limpieza agresiva de memoria después "
               "de cada consulta, y reintentos automáticos en caso de error. "
               "Soluciona el problema de degradación después de varios usos"),
    "3.9.13": ("NUEVA FUNCIONALIDAD: Página de detalle del producto - Al hacer clic "
               "en imágenes similares se abre página completa con toda la metadata, "
               "precios, stock, talles, etc. Mejora UX para clientes demo"),
    "3.9.12": ("LIMPIEZA CÓDIGO: Eliminados archivos redundantes (app.py, "
               "app_railway_production.py, production_server.py, scripts de inicio, "
               "diagnostico_gorras.py, gunicorn.conf.py). Proyecto más limpio y mantenible"),
    "3.9.11": ("ARQUITECTURA UNIFICADA: Eliminado app_railway.py redundante. "
               "Ahora app_simple.py maneja tanto desarrollo como producción Railway. "
               "Sin duplicación de código - principio DRY aplicado correctamente"),
    "3.9.10": ("CRITICAL RAILWAY FIX: Eliminado __pycache__ + rebuild forzado "
               "Railway DEBE usar classify_query_image() actualizada v3.9.10"),
    "3.9.9": ("FORCE COMPLETE DEPLOY: Rebuild completo Railway - asegurar "
              "que use código actualizado con mejoras de detección de hallucinations"),
    "3.9.8": ("FORCE DEPLOY: Forzar redeploy Railway con últimas mejoras "
              "en get_general_image_description() y classify_query_image()"),
    "3.9.7": ("FIX CLIP HALLUCINATIONS: Reordenados prompts priorizando "
              "categorías generales (texto, personas, etc.) antes que comerciales "
              "para evitar detección incorrecta de ropa en imágenes no comerciales"),
    "3.9.6": ("FIX EXIF: Corregir import error + detectión automática "
              "por dimensiones como fallback para móviles"),
    "3.9.5": ("DEBUG: Logs detallados para diagnosticar problema "
              "de orientación EXIF en móviles"),
    "3.9.4": ("FIX EXIF: Evitar doble corrección orientación - solo aplicar "
              "en archivos, no en objetos Image ya procesados"),
    "3.9.3": ("NUEVA FUNCIONALIDAD: Corrección automática de orientación "
              "EXIF para imágenes de móviles (rotación 90°)"),
    "3.9.2": ("FIX RUTAS IMÁGENES: Normalizar separadores \\ a / antes "
              "de basename() para compatibilidad Linux/Windows"),
    "3.9.1": ("FIX COMPLETO JSON: Convertir float32 en calculate_similarity "
              "y results para evitar errores serialización"),
    "3.9.0": ("FIX JSON SERIALIZATION: Convertir float32 PyTorch a float "
              "Python para evitar error 'not JSON serializable'"),
    "3.8.9": ("FIX CRÍTICO CATEGORÍAS: Corregido bucle classifications + "
              "generadas product_classifications.json para detección "
              "de productos"),
    "3.8.8": ("FIX DETECCIÓN CATEGORÍAS: Mejorada lógica para detectar "
              "'camisa' en 'camisa con botones y cuello'"),
    "3.8.7": ("FIX COMPATIBILIDAD: Removido half precision problemático + "
              "estado de modelo corregido"),
    "3.8.6": ("CORRECCIÓN CRÍTICA: RN50 (244MB) en lugar de ViT-B/32 "
              "(338MB) - Error de tamaños de modelos"),
    "3.8.5": ("OPTIMIZACIÓN MEMORIA: Sistema optimizado para 512MB RAM "
              "con lazy loading y garbage collection"),
    "3.8.0": ("DETECCIÓN AMPLIADA: Agregadas categorías no comercializadas "
              "para correcta identificación"),
    "3.7.0": ("ENFOQUE SIMPLIFICADO: Verificación genérica de categorías "
              "comercializadas vs no comercializadas")
}


def show_version_info():
    """Mostrar información de versión"""
    pass


# Configuración de la aplicación Flask (lazy loading de dependencias)
def create_app():
    """Crear aplicación Flask con lazy loading"""
    if not lazy_import_heavy_deps():
        raise Exception("No se pudieron cargar las dependencias")
    
    app = Flask(__name__)
    app.config['CATALOGO_FOLDER'] = 'catalogo'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    app.config['SECRET_KEY'] = os.environ.get(
        'SECRET_KEY', 'clip-demo-secret-key-2025')
    
    return app


# Crear app
app = create_app()

# Importar configuración centralizada de categorías
from config.categories import get_clip_categories, get_commercial_categories, is_commercial_category  # noqa: E401

# Importar función centralizada de búsqueda
from core.search_engine import find_similar_images  # noqa: E401

# Configuración de autenticación
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Por favor, inicia sesión para acceder.'

# Configuración de rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per day", "200 per hour", "50 per minute"]  # Límites más flexibles
)
limiter.init_app(app)

# Usuarios demo
DEMO_USERS = {
    'demo': 'clipdemo2025',
    'cliente': 'visual2025',
    'admin': 'clipadmin2025'
}


class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.username = username


@login_manager.user_loader
def load_user(user_id):
    if user_id in DEMO_USERS:
        return User(user_id)
    return None


# Variables globales
model = None
preprocess = None
device = None
catalog_embeddings = {}


def ensure_model_loaded():
    """Asegurar que el modelo esté cargado (lazy loading)"""
    global model, preprocess, device
    if model is None:
        load_clip_model()
    return model is not None


def load_clip_model():
    """Cargar el modelo CLIP con lazy loading y optimizaciones de memoria"""
    global model, preprocess, device
    
    # Lazy import de dependencias pesadas
    if not lazy_import_heavy_deps():
        return None, None
    
    try:
        # Configurar dispositivo (forzar CPU para ahorrar memoria)
        device = "cpu"  # Forzar CPU para 512MB RAM
        
        # Usar RN50 que sabemos que es más pequeño
        model, preprocess = clip.load("RN50", device=device)
        
        # Optimizaciones de memoria (sin half precision para compatibilidad)
        if hasattr(model, 'eval'):
            model.eval()
        
        # Forzar garbage collection agresivo
        import gc
        gc.collect()
        
        return model, preprocess
        
    except Exception:
        import traceback
        traceback.print_exc()
        return None, None


def get_image_embedding(image_input):
    """Generar embedding para una imagen - acepta path o objeto PIL Image"""
    global model, preprocess, device
    
    # Asegurar que el modelo esté cargado
    if not ensure_model_loaded():
        raise Exception("No se pudo cargar el modelo CLIP")
    
    try:
        # Determinar si es un path o un objeto Image
        if isinstance(image_input, str):
            image = Image.open(image_input)
            # Corregir orientación EXIF solo para archivos
            image = fix_image_orientation(image)
        else:
            image = image_input
            
        image = image.convert('RGB')
        
        # Redimensionar imagen para optimizar memoria
        max_size = 224  # Tamaño óptimo para RN50
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Preprocesar
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Generar embedding
        with torch.no_grad():
            image_tensor = image_tensor.float()
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten().astype(np.float32)
        
        return embedding
        
    except Exception as e:
        print(f"❌ Error generando embedding: {e}")
        # Limpieza básica en caso de error
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None


def load_catalog_embeddings():
    """Cargar embeddings del catálogo desde archivo"""
    global catalog_embeddings
    embeddings_file = "catalogo/embeddings.json"
    try:
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        
        catalog_embeddings = {}
        for filename, embedding_list in embeddings_data.items():
            catalog_embeddings[filename] = np.array(
                embedding_list, dtype=np.float32)
        
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def calculate_similarity(embedding1, embedding2):
    """Calcular similitud coseno entre dos embeddings"""
    similarity = (np.dot(embedding1, embedding2) /
                  (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    return float(similarity)  # Convertir a float Python para JSON


def fix_image_orientation(image):
    """Corregir orientación de imagen basándose en datos EXIF
    (especialmente para móviles)"""
    try:
        # Método 1: Intentar usar EXIF
        try:
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(274, 1)  # Default a 1 si no existe
                
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
                
                return image
        except Exception:
            pass
        
        # Método 2: Detección automática por dimensiones (FALLBACK)
        width, height = image.size
        # Imagen grande y horizontal mal orientada
        if width > height and width > 3000:
            image = image.rotate(90, expand=True)
            
    except Exception:
        pass
    
    return image


def get_general_image_description(image):
    """
    Obtener descripción LIBRE usando las categorías GOODY + categorías generales
    CLIP intentará detectar primero nuestros productos comerciales
    """
    try:
        # Tokenizar la imagen
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # CATEGORÍAS GENERALES - Para contenido no comercial (PRIMERO para evitar confusión)
        general_prompts = [
            "a photo of text and writing",
            "a photo of documents and papers",
            "a photo of written text and letters",
            "a photo of newspaper or magazine",
            "a photo of people and faces",
            "a photo of animals and pets",
            "a photo of vehicles and cars",
            "a photo of food and meals",
            "a photo of technology devices",
            "a photo of buildings and architecture",
            "a photo of nature and landscapes",
            "a photo of furniture and objects",
            "a photo of tools and equipment"
        ]
        
        # CATEGORÍAS COMERCIALES GOODY - Usar directamente nuestras 12 categorías
        commercial_categories = get_commercial_categories()  # Las 12 categorías oficiales GOODY
        
        # Convertir a prompts específicos para CLIP
        commercial_prompts = []
        for category in commercial_categories:
            # Usar cada categoría GOODY como prompt directo
            commercial_prompts.append(f"a photo of {category}")
        
        print(f"📋 Usando {len(general_prompts)} categorías generales + {len(commercial_prompts)} categorías comerciales GOODY")
        
        # Combinar prompts: PRIMERO generales, LUEGO comerciales
        all_prompts = general_prompts + commercial_prompts
        
        # Tokenizar prompts
        text_tokens = clip.tokenize(all_prompts).to(device)
        
        with torch.no_grad():
            # Obtener embeddings
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            
            # Normalizar
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calcular similitudes
            similarities = (image_features @ text_features.T).squeeze(0)
            
            # Obtener los mejores matches
            top_k = min(5, len(similarities))
            top_similarities = torch.topk(similarities, top_k)
            
            # Debug: mostrar top 5 descripciones
            print(f"🔍 Top {top_k} detecciones de CLIP:")
            descriptions = []
            for i, (sim_idx, sim_val) in enumerate(zip(top_similarities.indices, top_similarities.values)):
                prompt = all_prompts[sim_idx.item()]
                confidence = float(sim_val.item())
                descriptions.append((prompt, confidence))
                
                # Marcar si es comercial o general (NUEVO ORDEN: generales primero)
                is_commercial = sim_idx.item() >= len(general_prompts)
                marker = "🏪" if is_commercial else "🌐"
                print(f"   {i+1}. {marker} {prompt}: {confidence:.3f} ({confidence*100:.1f}%)")
            
            # Analizar la mejor descripción
            best_prompt = descriptions[0][0]
            best_confidence = descriptions[0][1]
            best_idx = top_similarities.indices[0].item()
            
            # Determinar si es comercial o general (NUEVO ORDEN: generales primero)
            is_commercial_detection = best_idx >= len(general_prompts)
            
            # Extraer descripción limpia
            if is_commercial_detection:
                # Es una categoría comercial - extraer de la categoría GOODY original
                commercial_idx = best_idx - len(general_prompts)
                original_category = commercial_categories[commercial_idx]
                # Extraer el primer término de la categoría GOODY
                clean_description = original_category.split(',')[0].strip()
                print(f"🏪 Categoría GOODY detectada: '{clean_description}'")
            else:
                # Es una categoría general
                if "text" in best_prompt.lower() or "document" in best_prompt.lower():
                    clean_description = "texto documento"
                elif "people" in best_prompt.lower() or "face" in best_prompt.lower():
                    clean_description = "persona"
                elif "animal" in best_prompt.lower() or "pet" in best_prompt.lower():
                    clean_description = "animal"
                elif "vehicle" in best_prompt.lower() or "car" in best_prompt.lower():
                    clean_description = "vehículo"
                elif "food" in best_prompt.lower() or "meal" in best_prompt.lower():
                    clean_description = "comida"
                else:
                    # Extraer palabra clave del prompt
                    clean_description = best_prompt.replace("a photo of", "").strip()
                print(f"🌐 Contenido general detectado: '{clean_description}'")
            
            print(f"🎯 Descripción extraída: '{clean_description}' {'(COMERCIAL)' if is_commercial_detection else '(GENERAL)'}")
            
            return clean_description, best_confidence
            
    except Exception as e:
        print(f"❌ Error en descripción: {e}")
        return "contenido no identificado", 0.0


def classify_query_image(image):
    """
    Flujo con descripción LIBRE de CLIP:
    1. CLIP describe libremente lo que ve (sin categorías forzadas)
    2. Analizamos si esa descripción coincide con nuestros productos comerciales
    3. Si coincide -> clasificación específica, si no -> informar descripción libre
    
    ⚠️ RAILWAY CRITICAL FIX v3.9.10 - CACHE CLEARED ⚠️
    """
    global model, preprocess, device
    try:
        print(f"��� RAILWAY v3.9.10 - CACHE CLEARED - NEW DEPLOYMENT ���")
        print(f"🆘 SI VES ESTO = RAILWAY ESTÁ USANDO EL CÓDIGO ACTUALIZADO 🆘")
        print(f"🔥 classify_query_image() v3.9.10 - ANTI-HALLUCINATIONS 🔥")
        
        # Determinar si es un path o un objeto Image
        if isinstance(image, str):
            image = Image.open(image)
            image = fix_image_orientation(image)
        else:
            image = image
            
        image = image.convert('RGB')
        print(f"📷 Imagen preparada: {image.size}")
        
        # PASO 1: Obtener descripción LIBRE de CLIP (sin limitaciones)
        print(f"🔍 PASO 1: Llamando get_general_image_description() VERSIÓN ACTUALIZADA")
        free_description, free_confidence = get_general_image_description(image)
        print(f"🔍 CLIP descripción libre: {free_description}")
        print(f"📊 Confianza descripción libre: {free_confidence:.3f} ({free_confidence*100:.1f}%)")
        
        # PASO 2: NOSOTROS analizamos si coincide con productos comerciales
        print(f"🔍 PASO 2: Verificando si es categoría comercial: '{free_description}'")
        if is_commercial_category(free_description):
            print(f"✅ Descripción libre coincide con categoría comercial")
            
            # PASO 3: Si es comercial, clasificar específicamente con categorías GOODY
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            commercial_categories = get_commercial_categories()
            
            text_tokens_commercial = clip.tokenize(commercial_categories).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features_commercial = model.encode_text(text_tokens_commercial)
                
                # Normalizar
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features_commercial = text_features_commercial / text_features_commercial.norm(dim=-1, keepdim=True)
                
                # Calcular similitudes con categorías comerciales específicas
                similarities_commercial = (image_features @ text_features_commercial.T).squeeze(0)
                
                # Obtener mejor match comercial específico
                best_commercial_idx = similarities_commercial.argmax().item()
                best_commercial_confidence = float(similarities_commercial[best_commercial_idx].item())
                best_commercial_category = commercial_categories[best_commercial_idx]
                
                # Debug: mostrar top 3 comerciales específicas
                top_similarities = torch.topk(similarities_commercial, min(3, len(similarities_commercial)))
                print(f"🔍 Top 3 categorías específicas GOODY:")
                for i, (sim_idx, sim_val) in enumerate(zip(top_similarities.indices, top_similarities.values)):
                    cat_name = commercial_categories[sim_idx.item()].split(',')[0].strip()
                    print(f"   {i+1}. {cat_name}: {sim_val.item():.3f} ({sim_val.item()*100:.1f}%)")
                
                if best_commercial_confidence >= 0.19:  # Umbral para categoría específica
                    category = best_commercial_category.split(',')[0].strip()
                    print(f"✅ Categoría comercial específica: {category} ({best_commercial_confidence*100:.1f}%)")
                    return category, best_commercial_confidence
                else:
                    print(f"❌ Confianza baja en categorías específicas (mejor: {best_commercial_confidence*100:.1f}%)")
                    # Es comercial pero no específico suficiente
                    return free_description, free_confidence
        
        # PASO 4: No es categoría comercial - usar descripción libre de CLIP
        print(f"🚫 Descripción libre NO coincide con categorías comerciales")
        print(f"🔄 RETORNANDO: NO_COMERCIAL:{free_description}")
        print(f"🎯 ESTE DEBE SER EL RESULTADO PARA VEHÍCULOS - v3.9.9")
        return f"NO_COMERCIAL:{free_description}", free_confidence
            
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en classify_query_image() v3.9.9: {e}")
        import traceback
        print(f"🔥 TRACEBACK v3.9.9: {traceback.format_exc()}")
        return None, 0.0

# ==================== RUTAS DE AUTENTICACIÓN ====================

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("20 per minute")  # Aumentado de 5 a 20 intentos por minuto
def login():
    """Página de login"""
    if current_user.is_authenticated:
        # Si el usuario es admin, redirigir al panel de administración
        if current_user.username == 'admin':
            return redirect(url_for('admin_panel'))
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in DEMO_USERS and DEMO_USERS[username] == password:
            user = User(username)
            login_user(user)
            
            # Redirigir según el rol del usuario
            if username == 'admin':
                return redirect(url_for('admin_panel'))
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Usuario o contraseña incorrectos')
            return render_template('login.html', error='Usuario o contraseña incorrectos')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Cerrar sesión"""
    logout_user()
    return redirect(url_for('login'))

# ==================== RUTAS PRINCIPALES ====================

@app.route('/')
@login_required
def index():
    """Página principal"""
    return render_template('index.html', user=current_user.username)

@app.route('/upload', methods=['POST'])
@login_required
@limiter.limit("100 per minute")  # Aumentado de 20 a 100 consultas por minuto
def upload_file():
    """Procesar imagen subida y encontrar similares - SIN GUARDAR EN DISCO"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        
        
        # Procesar imagen directamente desde memoria (SIN GUARDAR)
        try:
            # Leer el contenido completo primero
            file_content = file.read()
            file_size = len(file_content)
            
            
            # Crear un stream desde el contenido
            import io
            image_stream = io.BytesIO(file_content)
            image = Image.open(image_stream)
            
            
            # Corregir orientación EXIF (especialmente importante para móviles)
            
            image = fix_image_orientation(image)
            image = image.convert('RGB')
            
            
            # Convertir imagen a base64 para mostrar en frontend
            import base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            uploaded_image_data = f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            
            return jsonify({'error': 'Error procesando imagen'}), 500
        
        # Procesar imagen
        query_embedding = get_image_embedding(image)
        
        if query_embedding is None:
            return jsonify({'error': 'Error procesando imagen'}), 500

        print(f"\n🔍 === ANÁLISIS DE IMAGEN ===")
        print(f"📁 Archivo subido: {file.filename}")
        print(f"📏 Tamaño: {file_size:,} bytes")
        print(f"🖼️ Dimensiones: {image.size}")

        # Clasificar automáticamente usando el nuevo flujo
        query_type, query_confidence = classify_query_image(image)
        
        print(f"🎯 Resultado clasificación: {query_type}")
        print(f"📊 Confianza: {query_confidence:.3f} ({query_confidence*100:.1f}%)")
        
        # Determinar si es comercial o no comercial
        is_commercial = query_type and not query_type.startswith("NO_COMERCIAL:")
        general_description = None
        
        if not is_commercial and query_type and query_type.startswith("NO_COMERCIAL:"):
            # Extraer la descripción de lo que CLIP detectó
            general_description = query_type.replace("NO_COMERCIAL:", "").strip()
            print(f"� Contenido NO COMERCIAL: {general_description}")
        elif is_commercial:
            print(f"✅ Categoría COMERCIAL: {query_type}")
        else:
            print(f"❓ No se detectó contenido claro")

        # Buscar imágenes similares usando función centralizada
        similar_images = find_similar_images(
            query_embedding, 
            top_k=3, 
            query_type=query_type if is_commercial else None,  # Solo buscar si es comercial
            query_confidence=query_confidence
        )
        
        # Verificar si es una categoría explícitamente no comercializada
        if similar_images == "CATEGORIA_NO_COMERCIALIZADA":
            print(f"🚫 Categoría NO comercializada detectada")
            similar_images = []  # Convertir a lista vacía para manejo uniforme
        elif similar_images == "CATEGORIA_NO_DETECTADA":
            print(f"❓ No se detectó categoría clara")
            similar_images = []  # Convertir a lista vacía para manejo uniforme
        
        print(f"🔎 Imágenes similares encontradas: {len(similar_images)}")
        for i, (filename_path, similarity) in enumerate(similar_images, 1):
            basename = os.path.basename(filename_path.replace('\\', '/'))
            print(f"   {i}. {basename} - Similitud: {similarity:.3f} ({similarity*100:.1f}%)")
        
        if not similar_images:
            print(f"❌ No se encontraron imágenes similares en el catálogo")        
        
        # Preparar respuesta
        results = []
        for filename_path, similarity in similar_images:
            # Asegurar que solo tengamos el nombre del archivo (sin path)
            # Manejar tanto \ (Windows) como / (Linux) separadores
            basename = os.path.basename(filename_path.replace('\\', '/'))
            sim_float = float(similarity)  # Convertir a float Python primero
            results.append({
                'filename': basename,
                'similarity': sim_float,
                'similarity_percent': round(sim_float * 100, 2)
            })
        
        # Determinar mensaje de estado
        status_message = None
        if not similar_images:
            if general_description:
                # Contenido no comercial - CLIP detectó algo pero no se comercializa
                status_message = f"🚫 CLIP detectó: '{general_description.title()}' - GOODY no comercializa este tipo de productos. Nuestras categorías disponibles son: DELANTAL, AMBO, CAMISA, CASACA, ZUECO, GORRO, CARDIGAN, BUZO, ZAPATO DAMA, CHALECO, CHAQUETA, REMERA."
            elif is_commercial and query_type:
                # Es categoría comercializada pero sin productos similares
                detected_item = query_type.split(',')[0].strip().title()
                status_message = f"⚠️ Se detectó: '{detected_item}' (categoría comercializada) pero no se encontraron productos similares en nuestro catálogo actual."
            else:
                # Confianza muy baja en todo
                status_message = f"❌ No se pudo identificar claramente el contenido de la imagen (confianza: {query_confidence*100:.1f}%). Asegúrate de subir una imagen clara de ropa profesional."
        
        print(f"💬 Mensaje de estado: {status_message}")
        print(f"🏁 === FIN ANÁLISIS ===\n")
        
        # Preparar el tipo detectado para mostrar
        display_type = "No detectado"
        if is_commercial and query_type:
            # Categoría comercializada detectada
            detected_item = query_type.split(',')[0].strip().title()
            display_type = f"✅ {detected_item}"
        elif general_description:
            # Usar descripción de lo que CLIP realmente detectó
            detected_item = general_description.split(',')[0].strip().title()
            display_type = f"🚫 {detected_item} (No comercializado)"
        
        response_data = {
            'uploaded_file': file.filename,  # Solo nombre, no se guarda
            'uploaded_image_data': uploaded_image_data,  # Imagen en base64
            'query_type': display_type,
            'query_confidence': round(query_confidence, 3),
            'similar_images': results,
            'status_message': status_message
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        # Limpieza básica en caso de error
        import gc
        gc.collect()
        print(f"❌ Error procesando consulta: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Estado del sistema - Sin auth para health checks"""
    # Determinar estado real del modelo
    model_status = "No cargado"
    model_really_loaded = False
    
    try:
        if model is not None:
            model_status = "Cargado y listo"
            model_really_loaded = True
        else:
            # Verificar si las dependencias están disponibles
            if lazy_import_heavy_deps():
                model_status = "Disponible (lazy loading)"
                model_really_loaded = False
            else:
                model_status = "Error - dependencias no disponibles"
                model_really_loaded = False
    except Exception as e:
        model_status = f"Error: {str(e)}"
        model_really_loaded = False
    
    return jsonify({
        'version': VERSION,
        'build_date': BUILD_DATE,
        'model_loaded': model_really_loaded,
        'model_status': model_status,
        'catalog_size': len(catalog_embeddings),
        'device': str(device) if device else "None"
    })

@app.route('/catalogo/<filename>')
@login_required
def catalog_file(filename):
    """Servir archivos del catálogo"""
    return send_from_directory(app.config['CATALOGO_FOLDER'], filename)

@app.route('/producto/<filename>')
@login_required
def product_detail(filename):
    """Mostrar detalles completos de un producto específico"""
    try:
        # Cargar metadata del producto
        metadata_dict = load_metadata()
        product_metadata = link_metadata_to_image(filename, metadata_dict)
        
        # Verificar si la imagen existe en el catálogo
        catalog_images = get_catalog_images()
        if filename not in catalog_images:
            flash('Producto no encontrado en el catálogo', 'error')
            return redirect(url_for('index'))
        
        # Obtener categorías comerciales para mostrar información adicional
        commercial_categories = get_commercial_categories()
        
        return render_template('product_detail.html',
                             user=current_user.username,
                             filename=filename,
                             metadata=product_metadata,
                             commercial_categories=commercial_categories,
                             has_metadata=filename in metadata_dict)
        
    except Exception as e:
        flash(f'Error al cargar detalles del producto: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/admin/generate-embeddings', methods=['GET', 'POST'])
def generate_embeddings_endpoint():
    """Generar embeddings del catálogo disponible - Sin auth para setup inicial"""
    try:
        # Asegurar que el modelo esté cargado
        if not ensure_model_loaded():
            return jsonify({'error': 'No se pudo cargar el modelo CLIP'}), 500
            
        catalog_path = app.config['CATALOGO_FOLDER']
        if not os.path.exists(catalog_path):
            return jsonify({'error': 'Directorio catálogo no encontrado'}), 404
            
        # Importar y ejecutar generación de embeddings
        import subprocess
        result = subprocess.run([sys.executable, 'generate_embeddings.py'], 
                               capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            # Recargar embeddings
            load_catalog_embeddings()
            return jsonify({
                'success': True,
                'message': f'Embeddings generados para {len(catalog_embeddings)} imágenes',
                'catalog_size': len(catalog_embeddings)
            })
        else:
            return jsonify({'error': f'Error generando embeddings: {result.stderr}'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def initialize_system():
    """Inicializar sistema con lazy loading"""
    show_version_info()
    
    # Probar que el modelo se puede cargar (pero no mantenerlo en memoria)
    test_success = False
    try:
        if lazy_import_heavy_deps():
            # Hacer una prueba rápida de carga sin mantener el modelo
            import torch
            import clip
            device_test = "cpu"
            model_test, _ = clip.load("RN50", device=device_test)
            if model_test is not None:
                test_success = True
                # Limpiar memoria inmediatamente
                del model_test
                import gc
                gc.collect()
    except Exception:
        pass
    
    # Cargar embeddings del catálogo (sin requerir modelo)
    load_catalog_embeddings()
    
    return True

# ==================== FUNCIONES DE METADATA ====================

def load_metadata():
    """Cargar metadata desde archivo JSON"""
    metadata_file = "metadata.json"
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Si no existe, crear estructura básica
        return {}
    except Exception as e:
        print(f"Error cargando metadata: {e}")
        return {}

def save_metadata(data):
    """Guardar metadata en archivo JSON"""
    metadata_file = "metadata.json"
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error guardando metadata: {e}")
        return False

def get_catalog_images():
    """Obtener lista de imágenes del catálogo"""
    catalog_path = app.config['CATALOGO_FOLDER']
    images = []
    
    if os.path.exists(catalog_path):
        for filename in os.listdir(catalog_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images.append(filename)
    
    return sorted(images)

def link_metadata_to_image(image_filename, metadata_dict):
    """Vincular metadata a una imagen específica por nombre de archivo"""
    if image_filename in metadata_dict:
        return metadata_dict[image_filename]
    else:
        # Crear metadata por defecto para nuevas imágenes
        return {
            'nombre': image_filename,
            'categoria': '',
            'tipo_prenda': '',
            'color': '',
            'material': '',
            'talle': [],
            'precio': 0.0,
            'stock': 0,
            'fecha_ingreso': '',
            'notas': ''
        }

def validate_metadata_fields(metadata):
    """Validar campos obligatorios de metadata"""
    required_fields = ['nombre', 'categoria', 'tipo_prenda']
    errors = []
    
    for field in required_fields:
        if not metadata.get(field, '').strip():
            errors.append(f"El campo '{field}' es obligatorio")
    
    # Validar precio (debe ser número positivo)
    try:
        precio = float(metadata.get('precio', 0))
        if precio < 0:
            errors.append("El precio debe ser un número positivo")
    except (ValueError, TypeError):
        errors.append("El precio debe ser un número válido")
    
    # Validar stock (debe ser número entero positivo)
    try:
        stock = int(metadata.get('stock', 0))
        if stock < 0:
            errors.append("El stock debe ser un número entero positivo")
    except (ValueError, TypeError):
        errors.append("El stock debe ser un número entero válido")
    
    return errors

# ==================== RUTAS DEL PANEL DE ADMINISTRACIÓN ====================

@app.route('/admin_panel')
@login_required
def admin_panel():
    """Panel principal de administración - Solo para admin"""
    if current_user.username != 'admin':
        flash('Acceso denegado. Solo administradores pueden acceder.')
        return redirect(url_for('index'))
    
    # Obtener estadísticas del sistema
    total_images = len(get_catalog_images())
    metadata = load_metadata()
    images_with_metadata = len(metadata)
    
    stats = {
        'total_images': total_images,
        'images_with_metadata': images_with_metadata,
        'images_without_metadata': total_images - images_with_metadata,
        'catalog_size': len(catalog_embeddings),
        'version': VERSION
    }
    
    return render_template('admin_panel.html', user=current_user.username, stats=stats)

@app.route('/admin_metadata')
@login_required
def admin_metadata():
    """Administración de metadata de productos - Solo para admin"""
    if current_user.username != 'admin':
        flash('Acceso denegado. Solo administradores pueden acceder.')
        return redirect(url_for('index'))
    
    # Obtener imágenes del catálogo y su metadata
    catalog_images = get_catalog_images()
    metadata_dict = load_metadata()
    
    # Crear lista de imágenes con su metadata vinculada
    images_data = []
    for image_filename in catalog_images:
        image_metadata = link_metadata_to_image(image_filename, metadata_dict)
        images_data.append({
            'filename': image_filename,
            'metadata': image_metadata
        })
    
    # Obtener categorías disponibles del sistema
    commercial_categories = get_commercial_categories()
    
    return render_template('admin_metadata.html', 
                         user=current_user.username,
                         images_data=images_data,
                         commercial_categories=commercial_categories)

@app.route('/admin_metadata/edit/<filename>', methods=['GET', 'POST'])
@login_required
def edit_metadata(filename):
    """Editar metadata de una imagen específica"""
    if current_user.username != 'admin':
        flash('Acceso denegado. Solo administradores pueden acceder.')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Procesar formulario de edición
        metadata_dict = load_metadata()
        
        # Obtener datos del formulario
        new_metadata = {
            'nombre': request.form.get('nombre', ''),
            'categoria': request.form.get('categoria', ''),
            'tipo_prenda': request.form.get('tipo_prenda', ''),
            'color': request.form.get('color', ''),
            'material': request.form.get('material', ''),
            'talle': request.form.getlist('talle'),  # Lista de talles
            'precio': float(request.form.get('precio', 0)),
            'stock': int(request.form.get('stock', 0)),
            'fecha_ingreso': request.form.get('fecha_ingreso', ''),
            'notas': request.form.get('notas', '')
        }
        
        # Validar campos
        validation_errors = validate_metadata_fields(new_metadata)
        
        if validation_errors:
            for error in validation_errors:
                flash(error, 'error')
        else:
            # Guardar metadata
            metadata_dict[filename] = new_metadata
            
            if save_metadata(metadata_dict):
                flash(f'Metadata actualizada exitosamente para {filename}', 'success')
                return redirect(url_for('admin_metadata'))
            else:
                flash('Error al guardar la metadata', 'error')
    
    # Mostrar formulario de edición
    metadata_dict = load_metadata()
    image_metadata = link_metadata_to_image(filename, metadata_dict)
    commercial_categories = get_commercial_categories()
    
    # Lista de talles disponibles
    available_sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', 'Único']
    
    return render_template('edit_metadata.html',
                         user=current_user.username,
                         filename=filename,
                         metadata=image_metadata,
                         commercial_categories=commercial_categories,
                         available_sizes=available_sizes)

@app.route('/admin_metadata/delete/<filename>', methods=['POST'])
@login_required
def delete_metadata(filename):
    """Eliminar metadata de una imagen (no la imagen física)"""
    if current_user.username != 'admin':
        flash('Acceso denegado. Solo administradores pueden acceder.')
        return redirect(url_for('index'))
    
    metadata_dict = load_metadata()
    
    if filename in metadata_dict:
        del metadata_dict[filename]
        
        if save_metadata(metadata_dict):
            flash(f'Metadata eliminada exitosamente para {filename}', 'success')
        else:
            flash('Error al eliminar la metadata', 'error')
    else:
        flash('La imagen no tiene metadata asociada', 'warning')
    
    return redirect(url_for('admin_metadata'))

if __name__ == '__main__':
    if initialize_system():
        # Puerto dinámico para despliegue en cloud (Render, Heroku, Railway, etc.)
        port = int(os.environ.get('PORT', 5000))
        host = '0.0.0.0'
        
        # Detectar si estamos en Railway (ambiente de producción)
        is_production = os.environ.get('RAILWAY_ENVIRONMENT_NAME') is not None
        
        if is_production:
            # Usar Waitress para producción (Railway)
            try:
                from waitress import serve
                print(f"🏭 Modo PRODUCCIÓN activado")
                print(f"🚀 CLIP Comparador configurado para Railway")
                print(f"🌐 Puerto: {port}")
                print(f"🔧 Modo: production")
                print(f"🌐 Iniciando servidor WSGI (Waitress) en {host}:{port}")
                print("✅ Servidor de PRODUCCIÓN - Sin warnings de development")
                
                # Iniciar servidor Waitress para Railway
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
                app.run(host=host, port=port, debug=False, threaded=True)
        else:
            # Usar Flask development server para desarrollo local
            print(f"🛠️  Modo DESARROLLO activado")
            print(f"🌐 Ejecutando en: http://127.0.0.1:{port}")
            app.run(host=host, port=port, debug=True)
    else:
        sys.exit(1)
