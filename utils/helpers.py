"""
Funciones de utilidad
"""

import os

def _get_display_category(basename, classifications):
    """Obtener categoría para mostrar en resultados"""
    if basename in classifications:
        return classifications[basename]['category'][:8].upper()
    else:
        # Inferir por nombre
        if "CAMISA" in basename.upper(): return "CAMISA"
        elif "CHAQUETA" in basename.upper(): return "CHAQUETA"
        elif "DELANTAL" in basename.upper(): return "DELANTAL"
        elif "BUZO" in basename.upper(): return "BUZO"
        elif "GORRO" in basename.upper(): return "GORRO"
        elif "CASACA" in basename.upper(): return "CASACA"
        elif "AMBO" in basename.upper(): return "AMBO"
        elif "CARDIGAN" in basename.upper(): return "CARDIGAN"
        elif "CHALECO" in basename.upper(): return "CHALECO"
        elif "CALZADO" in basename.upper(): return "CALZADO"
        elif "ZAPATO" in basename.upper(): return "CALZADO"
        elif "REMERA" in basename.upper(): return "REMERA"
        else: return "OTRO"

def initialize_system():
    """Inicializar sistema con lazy loading"""
    from config.settings import show_version_info, lazy_import_heavy_deps
    from core.classification import load_catalog_embeddings
    from models.clip_model import ensure_model_loaded
    
    # Mostrar información de versión
    show_version_info()
    
    # Cargar modelo CLIP al inicio
    print("Cargando modelo CLIP...")
    if ensure_model_loaded():
        print("✅ Modelo CLIP cargado correctamente")
    else:
        print("❌ Error cargando modelo CLIP")
    
    # Cargar embeddings del catálogo
    print("Cargando embeddings del catálogo...")
    if load_catalog_embeddings():
        print("✅ Embeddings del catálogo cargados correctamente")
    else:
        print("❌ Error cargando embeddings del catálogo")
    
    return True