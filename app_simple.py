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
VERSION = "3.9.9"
BUILD_DATE = "2025-10-06"
CHANGES_LOG = {
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
    default_limits=["200 per day", "50 per hour", "10 per minute"]
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
            # (no para objetos ya procesados)
            image = fix_image_orientation(image)
        else:
            image = image_input
            
        image = image.convert('RGB')
        
        # Redimensionar imagen agresivamente para ahorrar memoria
        max_size = 224  # Mínimo posible para ViT-B/32
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Preprocesar (batch size 1 para memoria mínima)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Liberar imagen original inmediatamente
        del image
        import gc
        gc.collect()
        
        # Generar embedding con optimizaciones de memoria EXTREMAS
        with torch.no_grad():
            # Asegurar compatibilidad de tipos
            # siempre usar float32 para estabilidad
            image_tensor = image_tensor.float()  # Forzar float32
            
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True)
            
            # Liberar tensor inmediatamente
            embedding = image_features.cpu().numpy().flatten().astype(
                np.float32)
            
            # Limpiar todo inmediatamente
            del image_tensor, image_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return embedding
        
    except Exception:
        # Limpiar memoria en caso de error
        import gc
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
    
    ⚠️ RAILWAY DEPLOY v3.9.9 - FORCE REBUILD COMPLETO ⚠️
    """
    global model, preprocess, device
    try:
        print(f"🚀🚀🚀 INICIANDO classify_query_image() v3.9.9 - RAILWAY FORCE DEPLOY 🚀🚀🚀")
        print(f"🔥 ESTA ES LA VERSIÓN ACTUALIZADA CON FIX HALLUCINATIONS 🔥")
        
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
@limiter.limit("5 per minute")
def login():
    """Página de login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in DEMO_USERS and DEMO_USERS[username] == password:
            user = User(username)
            login_user(user)
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
@limiter.limit("20 per minute")
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

if __name__ == '__main__':
    if initialize_system():
        # Puerto dinámico para despliegue en cloud (Render, Heroku, etc.)
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        sys.exit(1)
