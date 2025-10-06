"""
Procesamiento de imágenes y generación de embeddings
"""

import os
from config.settings import lazy_import_heavy_deps
from models.clip_model import ensure_model_loaded, get_model, get_preprocess, get_device

def get_image_embedding(image_input):
    """Generar embedding para una imagen - acepta path o objeto PIL Image"""
    # Asegurar que el modelo esté cargado
    if not ensure_model_loaded():
        raise Exception("No se pudo cargar el modelo CLIP")
    
    # Lazy import
    if not lazy_import_heavy_deps():
        raise Exception("No se pudieron cargar las dependencias")
    
    import torch
    import numpy as np
    from PIL import Image
    
    model = get_model()
    preprocess = get_preprocess()
    device = get_device()
    
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