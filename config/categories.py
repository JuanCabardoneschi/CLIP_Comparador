"""
Configuración centralizada de categorías GOODY
"""

# CATEGORÍAS OFICIALES GOODY - 12 CATEGORÍAS COMERCIALIZADAS
CATEGORIAS_OFICIALES = [
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

# CATEGORÍAS PARA CLASIFICACIÓN CLIP (con términos en español e inglés)
CLIP_CATEGORIES = [
    # ===== CATEGORÍAS COMERCIALIZADAS POR GOODY =====
    "delantal de trabajo con pechera, mandil profesional con tirantes, apron with straps",
    "ambo vestir hombre y dama, uniforme médico scrubs, medical uniform set",
    "camisa hombre y dama, blusa formal de trabajo, dress shirt with collar",
    "casaca de chef profesional, chaqueta de cocina blanca, chef jacket with buttons",
    "zueco profesional de trabajo, calzado antideslizante, work clogs shoes",
    "gorro de chef blanco, gorra negra con visera, boina calada profesional, cap hat work",
    "cardigan hombre y dama, chaleco con botones, cardigan sweater",
    "buzo cerrado con capucha, sudadera de trabajo, hoodie sweatshirt",
    "zapato dama profesional, calzado femenino de trabajo, women work shoes",
    "chaleco dama y hombre, vest sin mangas profesional, sleeveless vest",
    "chaqueta profesional cerrada, jacket campera de trabajo, work jacket",
    "remera casual polo, camiseta sin botones, t-shirt cotton casual",
    # ===== CATEGORÍAS NO COMERCIALIZADAS (PARA DETECCIÓN Y RECHAZO) =====
    "pantalón largo, jean de trabajo, pants trousers long legs black",
    "short bermuda corto, pantalón corto de verano, summer shorts",
    "falda de vestir, pollera profesional, skirt for women",
    "vestido de trabajo, dress for women, ropa femenina formal",
    "corbata formal, tie necktie professional, formal tie"
]

# PALABRAS CLAVE PARA FILTROS DE CATEGORÍAS COMERCIALIZADAS
CATEGORIAS_COMERCIALIZADAS_KEYWORDS = [
    # DELANTAL
    "delantal", "pechera", "mandil", "apron",
    # AMBO VESTIR HOMBRE – DAMA
    "ambo", "uniforme médico", "scrubs", "medical uniform",
    # CAMISAS HOMBRE- DAMA
    "camisa", "shirt", "blusa", "dress shirt",
    # CASACAS
    "casaca", "chef jacket", "chaqueta de cocina",
    # ZUECOS
    "zueco", "clogs", "calzado profesional",
    # GORROS – GORRAS
    "gorro", "gorra", "cap", "hat", "boina",
    # CARDIGAN HOMBRE – DAMA
    "cardigan", "chaleco con botones", "sweater",
    # BUZOS
    "buzo", "hoodie", "sudadera", "sweatshirt",
    # ZAPATO DAMA
    "zapato dama", "calzado femenino", "women shoes",
    # CHALECO DAMA- HOMBRE
    "chaleco", "vest", "sin mangas",
    # CHAQUETAS
    "chaqueta", "jacket", "campera",
    # REMERAS
    "remera", "polo", "camiseta", "t-shirt"
]

# CATEGORÍAS NO COMERCIALIZADAS - GENERAN MENSAJE AMIGABLE
CATEGORIAS_NO_COMERCIALIZADAS = [
    "vestido", "dress", "falda", "skirt", 
    "pantalón", "pants", "jeans", "trousers",
    "corbata", "tie", "cinturón", "belt",
    "medias", "socks", "ropa interior", "underwear",
    "short", "bermuda", "shorts"
]

def get_commercial_categories():
    """Obtener lista de categorías comercializadas"""
    return CATEGORIAS_COMERCIALIZADAS_KEYWORDS.copy()

def get_non_commercial_categories():
    """Obtener lista de categorías no comercializadas"""
    return CATEGORIAS_NO_COMERCIALIZADAS.copy()

def get_clip_categories():
    """Obtener categorías para clasificación CLIP"""
    return CLIP_CATEGORIES.copy()

def is_commercial_category(query_text):
    """
    Verificar si un texto corresponde a una categoría comercializada
    """
    if not query_text:
        return False
    
    query_lower = query_text.lower()
    return any(categoria in query_lower for categoria in CATEGORIAS_COMERCIALIZADAS_KEYWORDS)

def is_non_commercial_category(query_text):
    """
    Verificar si un texto corresponde a una categoría NO comercializada
    """
    if not query_text:
        return False
    
    query_lower = query_text.lower()
    return any(categoria in query_lower for categoria in CATEGORIAS_NO_COMERCIALIZADAS)