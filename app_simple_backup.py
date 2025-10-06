"""
CLIP Comparador - Versión 3.8.1 - OPTIMIZADO PARA 512MB RAM
Sistema de búsqueda visual inteligente con modelo RN50 optimizado
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Importaciones optimizadas para 512MB RAM - LAZY LOADING
import os
import sys
import json
from datetime import datetime
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
    global LoginManager, UserMixin, login_user, login_required, logout_user, current_user
    global Limiter, get_remote_address
    
    try:
        import torch
        import clip  
        import numpy as np
        from PIL import Image
        from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
        from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        
        return True
    except ImportError:
        return False

# 🏷️ Sistema de Versioning Automático
VERSION = "3.9.6"
BUILD_DATE = "2025-10-02"
CHANGES_LOG = {
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
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clip-demo-secret-key-2025')
    
    return app

# Crear app
app = create_app()

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
        
    except Exception as e:
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
            # Corregir orientación EXIF solo para archivos (no para objetos ya procesados)
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
            # Asegurar compatibilidad de tipos - siempre usar float32 para estabilidad
            image_tensor = image_tensor.float()  # Forzar float32 para compatibilidad
            
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Liberar tensor inmediatamente
            embedding = image_features.cpu().numpy().flatten().astype(np.float32)
            
            # Limpiar todo inmediatamente
            del image_tensor, image_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return embedding
        
    except Exception as e:
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
            catalog_embeddings[filename] = np.array(embedding_list, dtype=np.float32)
        
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False

def calculate_similarity(embedding1, embedding2):
    """Calcular similitud coseno entre dos embeddings"""
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return float(similarity)  # Convertir a float Python para JSON serialization

def fix_image_orientation(image):
    """Corregir orientación de imagen basándose en datos EXIF (especialmente para móviles)"""
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
        if width > height and width > 3000:  # Imagen grande y horizontal mal orientada
            image = image.rotate(90, expand=True)
            
    except Exception:
        pass
    
    return image

def classify_query_image(image_input):
    """Clasificar imagen usando CLIP text embeddings - acepta path o objeto PIL Image"""
    global model, preprocess, device
    try:
        # Determinar si es un path o un objeto Image
        if isinstance(image_input, str):
            image = Image.open(image_input)
            # Corregir orientación EXIF solo para archivos (no para objetos ya procesados)
            image = fix_image_orientation(image)
        else:
            image = image_input
            
        image = image.convert('RGB')
            
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Categorías UNIFICADAS - 12 CATEGORÍAS DEL SISTEMA + CATEGORÍAS NO COMERCIALIZADAS
        categories = [
            # ===== CATEGORÍAS COMERCIALIZADAS POR GOODY =====
            "buzo cerrado con capucha, sudadera gruesa de trabajo, hoodie with zipper",
            "camisa con botones y cuello, blusa formal de trabajo, dress shirt with collar",
            "gorro de chef, gorra profesional con visera, work cap with brim, boina calada",
            "chaqueta cerrada profesional, jacket with zipper, campera de trabajo",
            "delantal de trabajo con pechera, mandil profesional con tirantes, apron with straps green",
            "ambo médico scrubs sanitario, uniforme hospitalario, medical uniform set",
            "casaca de chef blanca, chaqueta de cocina profesional, chef jacket with buttons",
            "zapato cerrado profesional, zueco de trabajo, calzado antideslizante, work shoes",
            "cardigan abierto con botones, chaleco sin mangas, vest without sleeves",
            "remera polo casual, camiseta deportiva sin botones, t-shirt casual cotton",
            "buzo frizado de trabajo, sudadera cerrada profesional, thick work sweatshirt",
            "conjunto deportivo casual, ropa de gimnasio, activewear clothing set",
            # ===== CATEGORÍAS NO COMERCIALIZADAS (PARA DETECCIÓN Y RECHAZO) =====
            "pantalón largo, jean de trabajo, pants trousers long legs black",
            "short bermuda corto, pantalón corto de verano, summer shorts",
            "falda de vestir, pollera profesional, skirt for women",
            "vestido de trabajo, dress for women, ropa femenina formal",
            "chaqueta de punto abierta, cardigan profesional, open front sweater"
        ]
        
        # Tokenizar categorías
        text_tokens = clip.tokenize(categories).to(device)
        
        with torch.no_grad():
            # Obtener embeddings de imagen y texto
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            
            # Normalizar
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calcular similitudes
            similarities = (image_features @ text_features.T).squeeze(0)
            
            # Obtener la categoría más probable
            best_match_idx = similarities.argmax().item()
            confidence = float(similarities[best_match_idx].item())  # Convertir a float de Python
            
            category = categories[best_match_idx].split(',')[0].strip()
            
            # UMBRAL MÍNIMO DE CONFIANZA - Si es muy bajo, no hay match claro
            MIN_CONFIDENCE_THRESHOLD = 0.20  # 20% mínimo para considerar válida la detección
            
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                return None, confidence
            
            return category, confidence
            
    except Exception:
        return None, 0.0

def find_similar_images(query_embedding, top_k=3, query_type=None, query_confidence=0.0):
    """
    VERSIÓN 3.3.0: MODOS INTELIGENTES COMPLETOS 
    Sistema específico para todas las 12 categorías profesionales
    """
    
    if not catalog_embeddings or query_embedding is None:
        return []

    # PASO 1: Calcular similitudes visuales base
    similarities = []
    for filename, catalog_embedding in catalog_embeddings.items():
        visual_similarity = calculate_similarity(query_embedding, catalog_embedding)
        similarities.append((filename, visual_similarity))

    # PASO 2: Cargar clasificaciones
    try:
        with open('catalogo/product_classifications.json', 'r', encoding='utf-8') as f:
            classifications = json.load(f)
    except:
        classifications = {}

    # PASO 3: SISTEMA DE MODOS INTELIGENTES COMPLETO CON VERIFICACIÓN DE DISPONIBILIDAD
    results = []
    
    # VERIFICAR SI HAY CATEGORÍA VÁLIDA DETECTADA
    if not query_type:
        return []
    
    # =================== CATEGORÍA: BUZOS (Evaluar ANTES que gorros) ===================
    if query_type and any(word in query_type.lower() for word in ["buzo", "hoodie", "sudadera", "frizado"]):
        # Verificar disponibilidad antes de buscar
        has_products, _ = _check_category_availability("buzo", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["BUZO", "HOODIE", "SUDADERA", "FRIZADO"], "🏃", 0.65, top_k)
    
    # =================== CATEGORÍA: CAMISAS ===================
    elif query_type and any(word in query_type.lower() for word in ["camisa", "shirt"]):
        has_products, _ = _check_category_availability("camisa", classifications)
        if not has_products:
            return []
        results = _filter_category_intelligent(similarities, classifications, 
                                               ["CAMISA", "SHIRT"], "👔", 0.65, top_k)
    
    # =================== CATEGORÍA: GORROS/GORRAS (Después de buzos) ===================
    elif query_type and any(word in query_type.lower() for word in ["gorro", "gorra", "cap", "hat", "boina"]) and "buzo" not in query_type.lower():
        
        has_products, _ = _check_category_availability("gorro", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["GORRO", "GORRA", "CAP", "HAT", "BOINA"], "🧢", 0.60, top_k)
    
    # =================== CATEGORÍA: CHAQUETAS ===================
    elif query_type and any(word in query_type.lower() for word in ["chaqueta", "jacket"]):
        
        has_products, _ = _check_category_availability("chaqueta", classifications)
        if not has_products:
            return []
        results = _filter_category_intelligent(similarities, classifications,
                                               ["CHAQUETA", "JACKET", "CAMISA", "CASACA", "CARDIGAN", "AMBO"], "🧥", 0.65, top_k)
    
    # =================== CATEGORÍA: DELANTALES ===================
    elif query_type and any(word in query_type.lower() for word in ["delantal", "mandil", "pechera"]):
        
        has_products, _ = _check_category_availability("delantal", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["DELANTAL", "MANDIL", "PECHERA", "JUMPER"], "🥽", 0.60, top_k)
    
    # =================== CATEGORÍA: AMBOS ===================
    elif query_type and any(word in query_type.lower() for word in ["ambo", "uniforme medico", "scrub"]):
        
        has_products, _ = _check_category_availability("ambo", classifications)
        if not has_products:
            return []
        results = _filter_category_intelligent(similarities, classifications,
                                               ["AMBO", "UNIFORME", "SCRUB", "CASACA"], "🏥", 0.65, top_k)
    
    # =================== CATEGORÍA: CASACAS ===================
    elif query_type and any(word in query_type.lower() for word in ["casaca", "blazer"]):
        
        has_products, _ = _check_category_availability("casaca", classifications)
        if not has_products:
            return []
        results = _filter_category_intelligent(similarities, classifications,
                                               ["CASACA", "BLAZER", "CHAQUETA", "CAMISA"], "🤵", 0.65, top_k)
    
    # =================== CATEGORÍA: CALZADO ===================
    elif query_type and any(word in query_type.lower() for word in ["zapato", "zueco", "calzado", "shoe"]):
        
        has_products, _ = _check_category_availability("zapato", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["ZAPATO", "ZUECO", "CALZADO", "SHOE"], "👟", 0.60, top_k)
    
    # =================== CATEGORÍA: CARDIGANS ===================
    elif query_type and any(word in query_type.lower() for word in ["cardigan", "chaleco", "vest"]):
        
        has_products, _ = _check_category_availability("cardigan", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["CARDIGAN", "CHALECO", "VEST"], "🧶", 0.65, top_k)
    
    # =================== CATEGORÍA: REMERAS ===================
    elif query_type and any(word in query_type.lower() for word in ["remera", "polo", "t-shirt", "playera"]):
        
        has_remeras, _ = _check_category_availability("remera", classifications)
        
        if has_remeras:
            # Si hay remeras, buscar solo remeras
            results = _filter_category_strict(similarities, classifications,
                                              ["REMERA", "POLO", "T-SHIRT", "PLAYERA"], "👕", 0.65, top_k)
        else:
            # Si no hay remeras, buscar en camisas (visualmente similares)
            has_camisas, _ = _check_category_availability("camisa", classifications)
            if has_camisas:
                results = _filter_category_strict(similarities, classifications,
                                                  ["CAMISA", "SHIRT", "BLUSA"], "👔", 0.60, top_k)
            else:
                return []
    
    # =================== MODO GENERAL (Para categorías no mapeadas específicamente) ===================
    else:
        results = _apply_general_search(similarities, classifications, query_type, query_confidence, top_k)

    # PASO 4: Verificar resultados finales Y CALIDAD DE SIMILITUD
    if not results:
        return []

    # VERIFICACIÓN DE SIMILITUD VISUAL MÍNIMA - Nueva función v3.7.0
    max_similarity = max(score for _, score in results)
    avg_similarity = sum(score for _, score in results) / len(results)
    
    MIN_VISUAL_SIMILARITY = 0.60  # 60% mínimo para considerar relevante
    
    if max_similarity < MIN_VISUAL_SIMILARITY:
        return []
    
    if avg_similarity < 0.45:  # Promedio muy bajo indica mala clasificación
        return []

    return results

def _filter_category_strict(similarities, classifications, keywords, emoji, threshold, top_k):
    """Filtrado estricto: solo productos de la categoría específica - MÁXIMO 3 RESULTADOS"""
    all_category_items = []  # Todos los items de la categoría
    
    for filename, visual_sim in similarities:
        basename = os.path.basename(filename)
        is_category = False
        
        # Verificar por nombre
        if any(word in basename.upper() for word in keywords):
            is_category = True
        # Verificar por clasificación
        elif basename in classifications:
            cat = classifications[basename]['category'].lower()
            if any(word.lower() in cat for word in keywords):
                is_category = True
        
        if is_category:
            all_category_items.append((filename, visual_sim))
    
    # SI NO HAY PRODUCTOS DE LA CATEGORÍA, RETORNAR VACÍO
    if not all_category_items:
        return []
    
    # Ordenar por similitud (mayor a menor)
    all_category_items.sort(key=lambda x: x[1], reverse=True)
    
    # SIEMPRE devolver los TOP 3 de la categoría (o los que haya si son menos)
    results = all_category_items[:top_k]
    
    # Mostrar los resultados seleccionados
    for filename, visual_sim in results:
        basename = os.path.basename(filename)
    
    return results

def _filter_category_intelligent(similarities, classifications, keywords, emoji, threshold, top_k):
    """Filtrado inteligente: categoría principal + productos relacionados con boost"""
    category_products = []
    related_products = []
    
    for filename, visual_sim in similarities:
        basename = os.path.basename(filename)
        is_main_category = False
        is_related = False
        
        # Verificar categoría principal (primer keyword)
        main_keyword = keywords[0]
        if main_keyword in basename.upper():
            is_main_category = True
        elif basename in classifications:
            cat = classifications[basename]['category'].lower()
            if main_keyword.lower() in cat:
                is_main_category = True
        
        # Verificar categorías relacionadas
        if not is_main_category:
            if any(word in basename.upper() for word in keywords[1:]):
                is_related = True
            elif basename in classifications:
                cat = classifications[basename]['category'].lower()
                if any(word.lower() in cat for word in keywords[1:]):
                    is_related = True
        
        if is_main_category:
            # Boost para categoría principal
            boosted_score = min(0.99, visual_sim * 1.15)
            category_products.append((filename, boosted_score))
        elif is_related:
            # Productos relacionados sin penalización excesiva
            related_products.append((filename, visual_sim))
    
    # Combinar resultados priorizando categoría principal
    all_products = category_products + related_products
    all_products.sort(key=lambda x: x[1], reverse=True)
    
    results = [(f, s) for f, s in all_products if s > threshold][:top_k]
    
    return results

def _check_category_availability(query_type, classifications):
    """Verificar si la categoría detectada corresponde a productos comercializados por GOODY"""
    if not query_type:
        return True, []
    
    query_lower = query_type.lower()
    available_products = []
    
    # Mapeo de categorías VÁLIDAS (productos que SÍ comercializa GOODY)
    goody_category_mappings = {
        'buzo': ['BUZO', 'BUZOS', 'SUDADERA', 'HOODIE', 'FRIZADO'],
        'camisa': ['CAMISA', 'CAMISAS', 'CAMISAS HOMBRE- DAMA', 'BLUSA', 'SHIRT'],
        'gorro': ['GORRO', 'GORROS', 'GORROS – GORRAS', 'GORRA', 'BOINA', 'CAP'],
        'chaqueta': ['CHAQUETA', 'CHAQUETAS', 'JACKET', 'CAMPERA'],
        'delantal': ['DELANTAL', 'MANDIL', 'APRON', 'PECHERA', 'JUMPER'],
        'ambo': ['AMBO', 'AMBO VESTIR HOMBRE – DAMA', 'SCRUBS', 'UNIFORME'],
        'casaca': ['CASACA', 'CASACAS', 'CHEF'],
        'zapato': ['ZAPATO', 'ZAPATO DAMA', 'ZUECO', 'ZUECOS', 'CALZADO', 'SHOE'],
        'cardigan': ['CARDIGAN', 'CARDIGAN HOMBRE – DAMA', 'CHALECO', 'CHALECO DAMA- HOMBRE', 'VEST'],
        'remera': ['REMERA', 'REMERAS', 'POLO', 'T-SHIRT', 'PLAYERA']
    }
    
    # Mapeo de categorías NO COMERCIALIZADAS (productos que NO vende GOODY)
    non_commercialized_mappings = {
        'pantalón': ['PANTALON', 'JEAN', 'PANTS', 'TROUSERS'],
        'short': ['SHORT', 'BERMUDA', 'SHORTS'],
        'falda': ['FALDA', 'POLLERA', 'SKIRT'],
        'vestido': ['VESTIDO', 'DRESS'],
    }
    
    # Verificar si la categoría detectada corresponde a productos comercializados
    category_found = False
    relevant_keywords = []
    is_non_commercialized = False
    
    # PASO 1: Verificar si es una categoría COMERCIALIZADA
    for goody_category, keywords in goody_category_mappings.items():
        # Buscar si alguna palabra clave de la categoría está en el query
        if any(keyword.lower() in query_lower for keyword in [goody_category] + keywords):
            category_found = True
            relevant_keywords = keywords
            
            break
    
    # PASO 2: Si no es comercializada, verificar si es NO COMERCIALIZADA
    if not category_found:
        for non_comm_category, keywords in non_commercialized_mappings.items():
            # Buscar si alguna palabra clave de categoría no comercializada está en el query
            if any(keyword.lower() in query_lower for keyword in [non_comm_category] + keywords):
                is_non_commercialized = True
                
                
                
                return False, []
                break
    
    # Si no encontramos la categoría en ningún mapeo, es categoría desconocida
    if not category_found and not is_non_commercialized:
        
        
        return False, []
    
    # Buscar productos disponibles en el catálogo para esta categoría válida
    for basename in classifications:
        # Verificar por nombre del archivo
        if any(keyword in basename.upper() for keyword in relevant_keywords):
            available_products.append(basename)
            continue
            
        # Verificar por categoría clasificada
        catalog_category = classifications[basename]['category'].upper()
        if any(keyword in catalog_category for keyword in relevant_keywords):
            available_products.append(basename)
    
    has_products = len(available_products) > 0
    
    return has_products, available_products

def _apply_general_search(similarities, classifications, query_type, query_confidence, top_k):
    """Búsqueda general con verificación de categorías disponibles"""
    
    # VERIFICAR SI ES UNA CATEGORÍA NO COMERCIALIZADA POR GOODY
    if query_type and query_confidence > 0.2:
        has_products, available_products = _check_category_availability(query_type, classifications)
        
        if not has_products:
            
            
            return []
    
    
    
    # Aplicar boost categórico ligero si hay buena confianza
    if query_type and query_confidence > 0.4:
        boosted_similarities = []
        for filename, visual_sim in similarities:
            basename = os.path.basename(filename)
            final_score = visual_sim
            
            if basename in classifications:
                catalog_category = classifications[basename]['category'].lower()
                if query_type.lower() in catalog_category or catalog_category in query_type.lower():
                    boost_factor = 1.1  # 10% boost ligero
                    final_score = min(0.99, visual_sim * boost_factor)
                    
            
            boosted_similarities.append((filename, final_score))
        
        similarities = boosted_similarities
    
    # Aplicar umbral general: solo productos con similitud > 65%
    similarities.sort(key=lambda x: x[1], reverse=True)
    results = [(f, s) for f, s in similarities if s > 0.65][:top_k]
    
    return results

def _get_display_category(basename, classifications):
    """Obtener categoría para mostrar en resultados"""
    if basename in classifications:
        return classifications[basename]['category'][:8].upper()
    else:
        # Inferir por nombre
        if "CAMISA" in basename.upper(): return "CAMISA"
        elif "CHAQUETA" in basename.upper(): return "CHAQUETA"
        elif "DELANTAL" in basename.upper(): return "DELANTAL"
        elif "AMBO" in basename.upper(): return "AMBO"
        elif "CASACA" in basename.upper(): return "CASACA"
        elif "GORRO" in basename.upper() or "GORRA" in basename.upper(): return "GORRO"
        elif "CARDIGAN" in basename.upper() or "CHALECO" in basename.upper(): return "CARDIGAN"
        elif "BUZO" in basename.upper(): return "BUZO"
        elif "REMERA" in basename.upper(): return "REMERA"
        elif "ZAPATO" in basename.upper() or "ZUECO" in basename.upper(): return "CALZADO"
        else: return "OTROS"

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
        
        # Clasificar automáticamente
        query_type, query_confidence = classify_query_image(image)
        
        # Buscar imágenes similares
        similar_images = find_similar_images(
            query_embedding, 
            top_k=3, 
            query_type=query_type, 
            query_confidence=query_confidence
        )
        
        
        
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
            if query_confidence < 0.20:  # Umbral muy bajo
                status_message = f"La imagen no coincide suficientemente con ninguna categoría conocida (confianza: {query_confidence*100:.1f}%)"
            else:
                # Verificar si la categoría detectada tiene productos disponibles
                try:
                    with open('catalogo/product_classifications.json', 'r', encoding='utf-8') as f:
                        classifications = json.load(f)
                    has_products, _ = _check_category_availability(query_type, classifications)
                    if not has_products:
                        status_message = f"GOODY no comercializa productos de la categoría '{query_type}'. Las categorías disponibles son: BUZO, CAMISA, GORRO, CHAQUETA, DELANTAL, AMBO, CASACA, CALZADO, CARDIGAN, REMERA"
                    else:
                        # Verificar si fue rechazado por similitud insuficiente
                        status_message = f"La imagen no muestra suficiente similitud visual con productos de la categoría '{query_type}'. Esto sugiere que podría ser un producto no comercializado por GOODY."
                except:
                    status_message = "No se encontraron productos similares en el catálogo"
        
        response_data = {
            'uploaded_file': file.filename,  # Solo nombre, no se guarda
            'uploaded_image_data': uploaded_image_data,  # Imagen en base64
            'query_type': query_type,
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
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        sys.exit(1)
