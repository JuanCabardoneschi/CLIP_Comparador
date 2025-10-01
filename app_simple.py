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

print("✅ Importaciones básicas OK")

# Variables globales para lazy loading
model = None
preprocess = None
device = None

# Variables globales
debug_delantales_info = []

def lazy_import_heavy_deps():
    """Importar dependencias pesadas solo cuando sea necesario"""
    global torch, clip, np, Image, Flask, render_template, request, jsonify
    global send_from_directory, redirect, url_for, flash, session
    global LoginManager, UserMixin, login_user, login_required, logout_user, current_user
    global Limiter, get_remote_address, secure_filename
    
    try:
        import torch
        print("✅ PyTorch importado correctamente")
        
        import clip  
        print("✅ CLIP importado correctamente")
        
        import numpy as np
        from PIL import Image
        from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
        from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        from werkzeug.utils import secure_filename
        
        print("✅ Todas las dependencias importadas correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error importando dependencias: {e}")
        return False

# 🏷️ Sistema de Versioning Automático
VERSION = "3.8.3"
BUILD_DATE = "2025-10-01"
CHANGES_LOG = {
    "3.8.3": "DIAGNÓSTICO MEJORADO: Verificación de modelo al inicio + logging detallado para debugging en producción",
    "3.8.2": "MODELO CLIP OPTIMIZADO: Cambiado de RN50x16 a RN50 para compatibilidad con 512MB RAM en producción",
    "3.8.1": "CORRECCIÓN CRÍTICA: Termina búsqueda al detectar categorías no comercializadas. No muestra productos irrelevantes.",
    "3.8.0": "DETECCIÓN AMPLIADA: Agregadas categorías no comercializadas (pantalón, short, falda, vestido) para correcta identificación y rechazo",
    "3.7.1": "UMBRAL DE SIMILITUD VISUAL: Rechaza resultados con similitud < 60% para evitar productos irrelevantes (ej: buzo para pantalón)",
    "3.7.0": "ENFOQUE SIMPLIFICADO: Verificación genérica de categorías comercializadas vs no comercializadas. Sin funciones específicas por producto.",
    "3.6.1": "DETECCIÓN MEJORADA DE PANTALONES: Compara directamente con descripciones de pantalón antes de clasificar",
    "3.6.0": "CATEGORÍAS NO DISPONIBLES: Detecta cuando GOODY no comercializa el tipo de producto (ej: pantalones) y muestra mensaje claro",
    "3.5.2": "UMBRAL CONFIANZA MÍNIMA: Si confianza < 20%, no muestra resultados - imagen no coincide con categorías",
    "3.5.1": "DELANTAL VERDE ESPECÍFICO: Restaurada categoría específica para delantales verdes que no se detectaban",
    "3.5.0": "CATEGORÍAS UNIFICADAS: Solo 12 categorías del sistema, eliminado mapeo confuso entre CLIP y modos",
    "3.4.2": "CATEGORÍA ESTRICTA: Si no hay productos de la categoría detectada, no muestra ningún resultado",
    "3.4.1": "TODAS LAS CATEGORÍAS CON 3 RESULTADOS: Cardigan y Remera ahora usan filtrado estricto para garantizar 3 resultados",
    "3.4.0": "SIEMPRE 3 RESULTADOS: Muestra los 3 mejores productos de cada categoría sin restricción de umbral",
    "3.3.1": "CORRECCIÓN MODO BUZO: Prioridad correcta y mejor detección de delantales verdes",
    "3.3.0": "MODOS INTELIGENTES COMPLETOS: Implementado sistema específico para todas las 12 categorías profesionales",
    "3.2.4": "MODO CHAQUETA INTELIGENTE: Busca solo chaquetas, camisas y productos relacionados, no gorras ni delantales",
    "3.2.3": "BÚSQUEDA CON UMBRAL: No fuerza resultados, solo muestra productos realmente similares, modo especial gorros",
    "3.2.1": "BÚSQUEDA VISUAL PRIORITARIA: Minimizado boost categórico, prioridad total a similitud visual",
    "3.2.0": "ALGORITMO MEJORADO: Búsqueda inteligente sin hardcode, boost categórico moderado, descriptores visuales específicos",
    "3.1.0": "CORRECCIÓN CLASIFICACIÓN: Mejorada diferenciación entre camisas y remeras, deshabilitado modo forzado",
    "3.0.0": "ACTUALIZACIÓN CATEGORÍAS: Implementadas 12 categorías profesionales GOODY oficiales",
}

def show_version_info():
    """Mostrar información de versión"""
    print(f"🚀 Iniciando CLIP Comparador...")
    print(f"📊 Versión: {VERSION} (Build: {BUILD_DATE})")
    print(f"🔄 Cambios en esta versión: {CHANGES_LOG[VERSION]}")

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
        print("🔄 Cargando modelo bajo demanda...")
        load_clip_model()
    return model is not None

def load_clip_model():
    """Cargar el modelo CLIP con lazy loading y optimizaciones de memoria"""
    global model, preprocess, device
    
    # Lazy import de dependencias pesadas
    if not lazy_import_heavy_deps():
        print("❌ Error: No se pudieron importar las dependencias")
        return None, None
    
    try:
        # Configurar dispositivo (forzar CPU para ahorrar memoria)
        device = "cpu"  # Forzar CPU para 512MB RAM
        print(f"🔄 Cargando modelo CLIP (RN50 - optimizado para 512MB RAM)...")
        
        # Usar modelo más pequeño y configuraciones de memoria
        model, preprocess = clip.load("RN50", device=device)
        
        # Optimizaciones de memoria
        if hasattr(model, 'eval'):
            model.eval()
        
        print(f"✅ Modelo CLIP RN50 cargado en: {device}")
        return model, preprocess
        
    except Exception as e:
        print(f"❌ Error cargando modelo CLIP: {str(e)}")
        print(f"❌ Tipo de error: {type(e).__name__}")
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
            print(f"🔄 Procesando imagen desde archivo: {os.path.basename(image_input)}")
            image = Image.open(image_input).convert('RGB')
        else:
            print(f"🔄 Procesando imagen desde memoria")
            image = image_input.convert('RGB')
        
        # Redimensionar imagen agresivamente para ahorrar memoria
        max_size = 224  # Reducido de 512 a 224 para ahorrar memoria
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        print(f"   📏 Redimensionada a: {image.size}")
        
        # Preprocesar (batch size 1 para memoria mínima)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Generar embedding con optimizaciones de memoria
        with torch.no_grad():
            # Usar autocast si está disponible para reducir memoria
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Liberar tensor de GPU inmediatamente
            embedding = image_features.cpu().numpy().flatten().astype(np.float32)  # float32 en lugar de float64
            
            # Limpiar cache de memoria
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"✅ Embedding generado - Norma: {np.linalg.norm(embedding):.4f}")
        return embedding
        
        return embedding
        
    except Exception as e:
        print(f"❌ Error procesando imagen: {str(e)}")
        return None

def load_catalog_embeddings():
    """Cargar embeddings del catálogo desde archivo"""
    global catalog_embeddings
    embeddings_file = "catalogo/embeddings.json"
    try:
        print("📁 Cargando embeddings del catálogo desde archivo...")
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        
        catalog_embeddings = {}
        for filename, embedding_list in embeddings_data.items():
            catalog_embeddings[filename] = np.array(embedding_list, dtype=np.float32)
        
        print(f"✅ Cargados {len(catalog_embeddings)} embeddings del catálogo")
        return True
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo {embeddings_file}")
        return False
    except Exception as e:
        print(f"❌ Error cargando embeddings: {str(e)}")
        return False

def calculate_similarity(embedding1, embedding2):
    """Calcular similitud coseno entre dos embeddings"""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def classify_query_image(image_input):
    """Clasificar imagen usando CLIP text embeddings - acepta path o objeto PIL Image"""
    global model, preprocess, device
    try:
        # Determinar si es un path o un objeto Image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
            
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
            confidence = similarities[best_match_idx].item()
            
            category = categories[best_match_idx].split(',')[0].strip()
            
            print(f"🔍 Imagen de consulta clasificada como: {category} (confianza: {confidence:.3f})")
            
            # UMBRAL MÍNIMO DE CONFIANZA - Si es muy bajo, no hay match claro
            MIN_CONFIDENCE_THRESHOLD = 0.20  # 20% mínimo para considerar válida la detección
            
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                print(f"⚠️ Confianza muy baja ({confidence*100:.1f}%) - No hay match claro con las categorías definidas")
                return None, confidence
            
            return category, confidence
            
    except Exception as e:
        print(f"❌ Error clasificando imagen: {str(e)}")
        return None, 0.0

def find_similar_images(query_embedding, top_k=3, query_type=None, query_confidence=0.0):
    """
    VERSIÓN 3.3.0: MODOS INTELIGENTES COMPLETOS 
    Sistema específico para todas las 12 categorías profesionales
    """
    print(f"\n🎯 ===== BÚSQUEDA INTELIGENTE v{VERSION} =====")
    print(f"📊 query_type: '{query_type}', confidence: {query_confidence:.3f}")
    
    if not catalog_embeddings or query_embedding is None:
        print("❌ ERROR: No hay embeddings del catálogo")
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
        print("❌ NO HAY CATEGORÍA VÁLIDA: La imagen no coincide suficientemente con ninguna de las 12 categorías definidas")
        print("🏷️ Categorías disponibles: BUZO, CAMISA, GORRO, CHAQUETA, DELANTAL, AMBO, CASACA, CALZADO, CARDIGAN, REMERA")
        return []
    
    # =================== CATEGORÍA: BUZOS (Evaluar ANTES que gorros) ===================
    if query_type and any(word in query_type.lower() for word in ["buzo", "hoodie", "sudadera", "frizado"]):
        print("🧠 MODO BUZO INTELIGENTE: Buscando buzos y sudaderas")
        # Verificar disponibilidad antes de buscar
        has_products, _ = _check_category_availability("buzo", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["BUZO", "HOODIE", "SUDADERA", "FRIZADO"], "🏃", 0.65, top_k)
    
    # =================== CATEGORÍA: CAMISAS ===================
    elif query_type and any(word in query_type.lower() for word in ["camisa", "shirt"]):
        print("🧠 MODO CAMISA INTELIGENTE: Priorizando camisas reales")
        has_products, _ = _check_category_availability("camisa", classifications)
        if not has_products:
            return []
        results = _filter_category_intelligent(similarities, classifications, 
                                               ["CAMISA", "SHIRT"], "👔", 0.65, top_k)
    
    # =================== CATEGORÍA: GORROS/GORRAS (Después de buzos) ===================
    elif query_type and any(word in query_type.lower() for word in ["gorro", "gorra", "cap", "hat", "boina"]) and "buzo" not in query_type.lower():
        print("🧠 MODO GORRO/GORRA INTELIGENTE: Buscando solo gorros/gorras")
        has_products, _ = _check_category_availability("gorro", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["GORRO", "GORRA", "CAP", "HAT", "BOINA"], "🧢", 0.60, top_k)
    
    # =================== CATEGORÍA: CHAQUETAS ===================
    elif query_type and any(word in query_type.lower() for word in ["chaqueta", "jacket"]):
        print("🧠 MODO CHAQUETA INTELIGENTE: Buscando chaquetas y productos similares")
        has_products, _ = _check_category_availability("chaqueta", classifications)
        if not has_products:
            return []
        results = _filter_category_intelligent(similarities, classifications,
                                               ["CHAQUETA", "JACKET", "CAMISA", "CASACA", "CARDIGAN", "AMBO"], "🧥", 0.65, top_k)
    
    # =================== CATEGORÍA: DELANTALES ===================
    elif query_type and any(word in query_type.lower() for word in ["delantal", "mandil", "pechera"]):
        print("🧠 MODO DELANTAL INTELIGENTE: Buscando solo delantales")
        has_products, _ = _check_category_availability("delantal", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["DELANTAL", "MANDIL", "PECHERA", "JUMPER"], "🥽", 0.60, top_k)
    
    # =================== CATEGORÍA: AMBOS ===================
    elif query_type and any(word in query_type.lower() for word in ["ambo", "uniforme medico", "scrub"]):
        print("🧠 MODO AMBO INTELIGENTE: Buscando uniformes médicos")
        has_products, _ = _check_category_availability("ambo", classifications)
        if not has_products:
            return []
        results = _filter_category_intelligent(similarities, classifications,
                                               ["AMBO", "UNIFORME", "SCRUB", "CASACA"], "🏥", 0.65, top_k)
    
    # =================== CATEGORÍA: CASACAS ===================
    elif query_type and any(word in query_type.lower() for word in ["casaca", "blazer"]):
        print("🧠 MODO CASACA INTELIGENTE: Buscando casacas profesionales")
        has_products, _ = _check_category_availability("casaca", classifications)
        if not has_products:
            return []
        results = _filter_category_intelligent(similarities, classifications,
                                               ["CASACA", "BLAZER", "CHAQUETA", "CAMISA"], "🤵", 0.65, top_k)
    
    # =================== CATEGORÍA: CALZADO ===================
    elif query_type and any(word in query_type.lower() for word in ["zapato", "zueco", "calzado", "shoe"]):
        print("🧠 MODO CALZADO INTELIGENTE: Buscando solo calzado")
        has_products, _ = _check_category_availability("zapato", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["ZAPATO", "ZUECO", "CALZADO", "SHOE"], "👟", 0.60, top_k)
    
    # =================== CATEGORÍA: CARDIGANS ===================
    elif query_type and any(word in query_type.lower() for word in ["cardigan", "chaleco", "vest"]):
        print("🧠 MODO CARDIGAN INTELIGENTE: Buscando cardigans y chalecos")
        has_products, _ = _check_category_availability("cardigan", classifications)
        if not has_products:
            return []
        results = _filter_category_strict(similarities, classifications,
                                          ["CARDIGAN", "CHALECO", "VEST"], "🧶", 0.65, top_k)
    
    # =================== CATEGORÍA: REMERAS ===================
    elif query_type and any(word in query_type.lower() for word in ["remera", "polo", "t-shirt", "playera"]):
        print("🧠 MODO REMERA INTELIGENTE: Buscando remeras y productos similares")
        has_remeras, _ = _check_category_availability("remera", classifications)
        
        if has_remeras:
            # Si hay remeras, buscar solo remeras
            results = _filter_category_strict(similarities, classifications,
                                              ["REMERA", "POLO", "T-SHIRT", "PLAYERA"], "👕", 0.65, top_k)
        else:
            # Si no hay remeras, buscar en camisas (visualmente similares)
            print("⚡ EXPANSIÓN INTELIGENTE: No hay remeras, buscando en camisas similares")
            has_camisas, _ = _check_category_availability("camisa", classifications)
            if has_camisas:
                results = _filter_category_strict(similarities, classifications,
                                                  ["CAMISA", "SHIRT", "BLUSA"], "👔", 0.60, top_k)
                print(f"🔄 BÚSQUEDA EXPANDIDA: Mostrando camisas como alternativa a remeras")
            else:
                return []
    
    # =================== MODO GENERAL (Para categorías no mapeadas específicamente) ===================
    else:
        print(f"🔍 MODO GENERAL: Búsqueda visual para categoría '{query_type}'")
        results = _apply_general_search(similarities, classifications, query_type, query_confidence, top_k)

    # PASO 4: Verificar resultados finales Y CALIDAD DE SIMILITUD
    if not results:
        print("❌ NO HAY RESULTADOS: No se encontraron productos de la categoría detectada en el catálogo")
        return []

    # VERIFICACIÓN DE SIMILITUD VISUAL MÍNIMA - Nueva función v3.7.0
    max_similarity = max(score for _, score in results)
    avg_similarity = sum(score for _, score in results) / len(results)
    
    MIN_VISUAL_SIMILARITY = 0.60  # 60% mínimo para considerar relevante
    
    if max_similarity < MIN_VISUAL_SIMILARITY:
        print(f"🚫 SIMILITUD INSUFICIENTE: Máxima similitud {max_similarity:.3f} ({max_similarity*100:.1f}%) < {MIN_VISUAL_SIMILARITY*100:.0f}% mínimo")
        print(f"💭 Los productos encontrados no son visualmente similares a la imagen consultada")
        print(f"🏷️ Esto sugiere que la imagen podría ser de un producto no comercializado por GOODY")
        return []
    
    if avg_similarity < 0.45:  # Promedio muy bajo indica mala clasificación
        print(f"⚠️ SIMILITUD PROMEDIO BAJA: {avg_similarity:.3f} ({avg_similarity*100:.1f}%) sugiere clasificación incorrecta")
        print(f"💭 La imagen podría ser de un producto no comercializado por GOODY")
        return []

    print(f"🎯 RESULTADOS FINALES ({len(results)} productos encontrados):")
    for i, (filename, score) in enumerate(results):
        basename = os.path.basename(filename)
        category = _get_display_category(basename, classifications)
        print(f"   {i+1}. {category:8s} {basename}: {score:.4f} ({score*100:.1f}%)")
    
    print(f"===== FIN BÚSQUEDA INTELIGENTE =====\n")
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
        print(f"   ❌ NO HAY PRODUCTOS de categoría {keywords[0]} en el catálogo")
        return []
    
    # Ordenar por similitud (mayor a menor)
    all_category_items.sort(key=lambda x: x[1], reverse=True)
    
    # MOSTRAR DEBUG PARA DELANTALES
    if keywords[0] == "DELANTAL":
        print(f"   🔍 DEBUG - TODOS LOS DELANTALES:")
        debug_info = []
        for filename, sim in all_category_items:
            basename = os.path.basename(filename)
            status = "✅" if sim > threshold else "❌"
            debug_line = f"      {status} {basename}: {sim:.3f} ({sim*100:.1f}%)"
            print(debug_line)
            debug_info.append(debug_line)
        
        # Guardar debug info globalmente
        global debug_delantales_info
        debug_delantales_info = debug_info
    
    # SIEMPRE devolver los TOP 3 de la categoría (o los que haya si son menos)
    results = all_category_items[:top_k]
    
    # Mostrar los resultados seleccionados
    for filename, visual_sim in results:
        basename = os.path.basename(filename)
        print(f"   {emoji} {basename}: {visual_sim:.3f}")
    
    return results
    if not results:
        print(f"   ⚠️ No se encontraron productos de esta categoría con similitud > {threshold}")
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
            print(f"   {emoji} PRINCIPAL: {basename} {visual_sim:.3f} -> {boosted_score:.3f}")
        elif is_related:
            # Productos relacionados sin penalización excesiva
            related_products.append((filename, visual_sim))
            print(f"   {emoji} RELACIONADO: {basename} {visual_sim:.3f}")
    
    # Combinar resultados priorizando categoría principal
    all_products = category_products + related_products
    all_products.sort(key=lambda x: x[1], reverse=True)
    
    results = [(f, s) for f, s in all_products if s > threshold][:top_k]
    if not results:
        print(f"   ⚠️ No se encontraron productos relacionados con similitud > {threshold}")
    return results

def _check_category_availability(query_type, classifications):
    """Verificar si la categoría detectada corresponde a productos comercializados por GOODY"""
    if not query_type:
        return True, []
    
    query_lower = query_type.lower()
    available_products = []
    
    # Mapeo de categorías VÁLIDAS (productos que SÍ comercializa GOODY)
    goody_category_mappings = {
        'buzo': ['BUZO', 'SUDADERA', 'HOODIE', 'FRIZADO'],
        'camisa': ['CAMISA', 'BLUSA', 'SHIRT'],
        'gorro': ['GORRO', 'GORRA', 'BOINA', 'CAP'],
        'chaqueta': ['CHAQUETA', 'JACKET', 'CAMPERA'],
        'delantal': ['DELANTAL', 'MANDIL', 'APRON', 'PECHERA', 'JUMPER'],
        'ambo': ['AMBO', 'SCRUBS', 'UNIFORME'],
        'casaca': ['CASACA', 'CHEF'],
        'zapato': ['ZAPATO', 'ZUECO', 'CALZADO', 'SHOE'],
        'cardigan': ['CARDIGAN', 'CHALECO', 'VEST'],
        'remera': ['REMERA', 'POLO', 'T-SHIRT', 'PLAYERA']
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
        if goody_category in query_lower:
            category_found = True
            relevant_keywords = keywords
            break
    
    # PASO 2: Si no es comercializada, verificar si es NO COMERCIALIZADA
    if not category_found:
        for non_comm_category, keywords in non_commercialized_mappings.items():
            if non_comm_category in query_lower:
                is_non_commercialized = True
                print(f"🚫 CATEGORÍA NO COMERCIALIZADA: '{query_type}' corresponde a '{non_comm_category.upper()}'")
                print(f"💬 GOODY no comercializa este tipo de productos ({non_comm_category})")
                print(f"🏷️ Categorías disponibles: BUZO, CAMISA, GORRO, CHAQUETA, DELANTAL, AMBO, CASACA, CALZADO, CARDIGAN, REMERA")
                return False, []
    
    # Si no encontramos la categoría en ningún mapeo, es categoría desconocida
    if not category_found and not is_non_commercialized:
        print(f"🚫 CATEGORÍA NO COMERCIALIZADA: '{query_type}' no corresponde a productos que comercializa GOODY")
        print(f"🏷️ Categorías disponibles: BUZO, CAMISA, GORRO, CHAQUETA, DELANTAL, AMBO, CASACA, CALZADO, CARDIGAN, REMERA")
        return False, []
    
    # Buscar productos disponibles en el catálogo para esta categoría válida
    for filename in classifications:
        basename = os.path.basename(filename)
        
        # Verificar por nombre del archivo
        if any(keyword in basename.upper() for keyword in relevant_keywords):
            available_products.append(basename)
            continue
            
        # Verificar por categoría clasificada
        catalog_category = classifications[basename]['category'].upper()
        if any(keyword in catalog_category for keyword in relevant_keywords):
            available_products.append(basename)
    
    has_products = len(available_products) > 0
    
    if not has_products:
        print(f"⚠️ CATEGORÍA VÁLIDA SIN STOCK: '{query_type}' es válida pero no hay productos en el catálogo")
    else:
        print(f"✅ CATEGORÍA DISPONIBLE: '{query_type}' - {len(available_products)} productos encontrados")
        
    return has_products, available_products
    
    # SEGUNDA VERIFICACIÓN: Mapeo de categorías detectadas a palabras clave del catálogo
    category_mappings = {
        'buzo': ['BUZO', 'SUDADERA', 'HOODIE', 'FRIZADO'],
        'camisa': ['CAMISA', 'BLUSA', 'SHIRT'],
        'gorro': ['GORRO', 'GORRA', 'BOINA', 'CAP'],
        'chaqueta': ['CHAQUETA', 'JACKET', 'CAMPERA'],
        'delantal': ['DELANTAL', 'MANDIL', 'APRON', 'PECHERA', 'JUMPER'],
        'ambo': ['AMBO', 'SCRUBS', 'UNIFORME'],
        'casaca': ['CASACA', 'CHEF'],
        'zapato': ['ZAPATO', 'ZUECO', 'CALZADO', 'SHOE'],
        'cardigan': ['CARDIGAN', 'CHALECO', 'VEST'],
        'remera': ['REMERA', 'POLO', 'T-SHIRT', 'PLAYERA']
    }
    
    # Encontrar palabras clave relevantes para la categoría detectada
    relevant_keywords = []
    for category, keywords in category_mappings.items():
        if category in query_lower:
            relevant_keywords = keywords
            break
    
    # Si no encontramos mapeo específico, usar la palabra detectada directamente
    if not relevant_keywords:
        relevant_keywords = [query_type.upper()]
    
    # TERCERA VERIFICACIÓN: Buscar productos que coincidan con las palabras clave
    for filename in classifications:
        basename = os.path.basename(filename)
        
        # Verificar por nombre del archivo
        if any(keyword in basename.upper() for keyword in relevant_keywords):
            available_products.append(basename)
            continue
            
        # Verificar por categoría clasificada
        catalog_category = classifications[basename]['category'].upper()
        if any(keyword in catalog_category for keyword in relevant_keywords):
            available_products.append(basename)
    
    has_products = len(available_products) > 0
    
    if not has_products:
        print(f"⚠️ CATEGORÍA SIN PRODUCTOS: '{query_type}' no tiene productos disponibles en el catálogo GOODY")
        print(f"   Keywords buscadas: {relevant_keywords}")
        
    return has_products, available_products

def _detect_semantic_mismatch(query_type, query_confidence):
    """Detectar si hay un desajuste semántico entre lo detectado y las categorías reales de GOODY"""
    
    # Lista de productos que claramente NO comercializa GOODY
    non_goody_products = [
        'pantalón', 'pantalon', 'pants', 'trousers', 'jean', 'denim',
        'short', 'bermuda', 'falda', 'skirt', 'vestido', 'dress',
        'corbata', 'tie', 'cinturón', 'belt', 'sombrero', 'hat',
        'medias', 'calcetines', 'socks', 'guantes', 'gloves',
        'mochila', 'backpack', 'bolso', 'bag'
    ]
    
    # Si la confianza es moderada (40-80%) y detectó una categoría válida de GOODY,
    # pero visualmente podría ser un producto no comercializado
    if 0.4 <= query_confidence <= 0.8:
        # Categorías que frecuentemente son "falsos positivos" para productos no comercializados
        false_positive_categories = [
            'zapato cerrado profesional',  # Frecuente para pantalones
            'cardigan abierto con botones',  # Frecuente para pantalones cargo
            'camisa con botones y cuello'   # Frecuente para pantalones formales
        ]
        
        if query_type and any(fp in query_type.lower() for fp in ['zapato', 'cardigan', 'camisa']):
            print(f"🤔 POSIBLE FALSO POSITIVO: '{query_type}' con confianza {query_confidence*100:.1f}%")
            print(f"   Podría ser un producto no comercializado clasificado incorrectamente")
            return True
    
    return False

def _apply_general_search(similarities, classifications, query_type, query_confidence, top_k):
    """Búsqueda general con verificación de desajustes semánticos"""
    
    # VERIFICAR SI ES UNA CATEGORÍA NO COMERCIALIZADA POR GOODY
    if query_type and query_confidence > 0.2:  # Cambiado de 0.4 a 0.2 para ser más estricto
        has_products, available_products = _check_category_availability(query_type, classifications)
        
        if not has_products:
            print(f"❌ CATEGORÍA '{query_type}' NO DISPONIBLE EN CATÁLOGO - BÚSQUEDA TERMINADA")
            print("💡 GOODY no comercializa este tipo de producto")
            return []  # TERMINAR AQUÍ - No continuar con búsqueda general
    
    # Solo continuar si la categoría ES comercializada por GOODY
    print(f"✅ CATEGORÍA '{query_type}' COMERCIALIZADA - Continuando búsqueda...")
    
    # VERIFICAR POSIBLES DESAJUSTES SEMÁNTICOS (falsos positivos)
    # Continuar con búsqueda normal si pasa las verificaciones
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
                    print(f"   ⬆️ BOOST: {basename} {visual_sim:.3f} -> {final_score:.3f}")
            
            boosted_similarities.append((filename, final_score))
        
        similarities = boosted_similarities
    
    # Aplicar umbral general: solo productos con similitud > 65%
    similarities.sort(key=lambda x: x[1], reverse=True)
    results = [(f, s) for f, s in similarities if s > 0.65][:top_k]
    
    if not results:
        print("   ⚠️ No se encontraron productos con similitud suficiente (> 65%)")
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
        print("📤 Recibida petición de upload")
        
        if 'file' not in request.files:
            print("❌ No se encontró campo 'file' en la petición")
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        print(f"📁 Archivo recibido: {file.filename}")
        
        # Procesar imagen directamente desde memoria (SIN GUARDAR)
        try:
            # Leer el contenido completo primero
            file_content = file.read()
            file_size = len(file_content)
            print(f"✅ Archivo leído - Tamaño: {file_size} bytes")
            
            # Crear un stream desde el contenido
            import io
            image_stream = io.BytesIO(file_content)
            image = Image.open(image_stream).convert('RGB')
            print(f"✅ Imagen cargada en memoria - Tamaño original: {image.size}")
            
            # Convertir imagen a base64 para mostrar en frontend
            import base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            uploaded_image_data = f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            print(f"❌ Error cargando imagen en memoria: {str(e)}")
            return jsonify({'error': 'Error procesando imagen'}), 500
        
        # Procesar imagen
        print(f"🔄 Procesando imagen subida: {file.filename}")
        query_embedding = get_image_embedding(image)
        
        if query_embedding is None:
            return jsonify({'error': 'Error procesando imagen'}), 500
        
        print(f"✅ Embedding generado - Shape: {query_embedding.shape}")
        
        # Clasificar automáticamente
        query_type, query_confidence = classify_query_image(image)
        print(f"🤖 MODO AUTO: Tipo auto-detectado: '{query_type}' (confianza: {query_confidence*100:.1f}%)")
        
        print(f"🎯 PARÁMETROS FINALES PARA BÚSQUEDA:")
        print(f"   query_type: '{query_type}'")
        print(f"   query_confidence: {query_confidence}")
        print(f"   embedding shape: {query_embedding.shape}")
        
        # Buscar imágenes similares
        print("🚀 LLAMANDO A find_similar_images...")
        similar_images = find_similar_images(
            query_embedding, 
            top_k=3, 
            query_type=query_type, 
            query_confidence=query_confidence
        )
        
        print(f"✅ find_similar_images completado, resultados: {len(similar_images)}")
        
        # Preparar respuesta
        results = []
        for filename_path, similarity in similar_images:
            basename = os.path.basename(filename_path)
            results.append({
                'filename': basename,
                'similarity': float(similarity),
                'similarity_percent': round(similarity * 100, 2)
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
            'status_message': status_message,
            'debug_delantales': debug_delantales_info if debug_delantales_info else None
        }
        
        print(f"✅ Procesamiento completo - {len(results)} imágenes similares encontradas")
        print(f"💾 OPTIMIZACIÓN: Imagen procesada en memoria, no se guardó en disco")
        if status_message:
            print(f"💬 Mensaje de estado: {status_message}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error en upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Estado del sistema - Sin auth para health checks"""
    # Intentar cargar el modelo si no está cargado para dar un estado más preciso
    model_status = "No cargado"
    try:
        if model is not None:
            model_status = "Cargado"
        else:
            # Intentar lazy loading para verificar si es posible cargar
            if lazy_import_heavy_deps():
                model_status = "Disponible (lazy loading)"
            else:
                model_status = "Error - dependencias no disponibles"
    except Exception as e:
        model_status = f"Error: {str(e)}"
    
    return jsonify({
        'version': VERSION,
        'build_date': BUILD_DATE,
        'model_loaded': model is not None,
        'model_status': model_status,
        'catalog_size': len(catalog_embeddings),
        'device': str(device) if device else "None"
    })

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    """Ruta obsoleta - las imágenes ya no se guardan"""
    return jsonify({'error': 'Las imágenes subidas ya no se guardan en disco'}), 404

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
    print("🔄 Verificando disponibilidad del modelo CLIP...")
    test_success = False
    try:
        if lazy_import_heavy_deps():
            # Hacer una prueba rápida de carga sin mantener el modelo
            import torch
            import clip
            device_test = "cpu"
            print("🔄 Prueba de carga del modelo RN50...")
            model_test, _ = clip.load("RN50", device=device_test)
            if model_test is not None:
                print("✅ Modelo RN50 verificado exitosamente")
                test_success = True
                # Limpiar memoria inmediatamente
                del model_test
                import gc
                gc.collect()
            else:
                print("❌ Error: Modelo no se pudo cargar")
        else:
            print("❌ Error: Dependencias no disponibles")
    except Exception as e:
        print(f"❌ Error verificando modelo: {str(e)}")
        print(f"❌ Tipo de error: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    if test_success:
        print("🔄 Configurando lazy loading - Modelo se cargará cuando sea necesario")
    else:
        print("⚠️ ADVERTENCIA: El modelo no se pudo verificar, pero continuando con lazy loading")
    
    # Cargar embeddings del catálogo (sin requerir modelo)
    if not load_catalog_embeddings():
        print("⚠️ Embeddings no encontrados - usar /admin/generate-embeddings después del deployment")
    
    print("✅ Sistema listo con optimizaciones de memoria!")
    print(f"📁 Catálogo: {len(catalog_embeddings)} imágenes")
    return True

if __name__ == '__main__':
    if initialize_system():
        # Puerto dinámico para despliegue en cloud (Render, Heroku, etc.)
        port = int(os.environ.get('PORT', 5000))
        print(f"🌐 Iniciando servidor web en puerto {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Error en la inicialización")
        sys.exit(1)