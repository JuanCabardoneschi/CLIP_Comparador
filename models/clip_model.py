"""
Gestión del modelo CLIP
"""

from config.settings import lazy_import_heavy_deps

# Variables globales
model = None
preprocess = None
device = None

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
        import clip
        
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

def get_model():
    """Obtener instancia del modelo"""
    return model

def get_preprocess():
    """Obtener función de preprocesamiento"""
    return preprocess

def get_device():
    """Obtener dispositivo configurado"""
    return device