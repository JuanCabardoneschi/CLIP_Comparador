"""
Clasificación de imágenes usando CLIP
"""

import json
import numpy as np
from config.settings import lazy_import_heavy_deps
from models.clip_model import ensure_model_loaded, get_model, get_preprocess, get_device
from core.image_processing import fix_image_orientation

# Variable global para este módulo
catalog_embeddings = {}

def classify_query_image(image_input):
    """Clasificar imagen usando CLIP text embeddings - acepta path o objeto PIL Image"""
    if not ensure_model_loaded():
        raise Exception("No se pudo cargar el modelo CLIP")
    
    # Lazy import
    if not lazy_import_heavy_deps():
        raise Exception("No se pudieron cargar las dependencias")
    
    import torch
    import clip
    from PIL import Image
    from config.categories import get_clip_categories
    
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
            
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Obtener categorías desde configuración centralizada
        categories = get_clip_categories()
        
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
            
            # Log detallado para categorías de gorras/gorros
            if any(keyword in category.lower() for keyword in ['gorro', 'gorra', 'boina', 'cap']):
                print(f"🧢 Categoría detectada: {category} con confianza {confidence:.3f}")
                # Mostrar top 3 categorías para debugging
                top3_indices = similarities.argsort(descending=True)[:3]
                for i, idx in enumerate(top3_indices):
                    cat_name = categories[idx].split(',')[0].strip()
                    conf = float(similarities[idx].item())
                    print(f"  #{i+1}: {cat_name} ({conf:.3f})")
            
            # UMBRAL MÍNIMO DE CONFIANZA - Si es muy bajo, no hay match claro
            MIN_CONFIDENCE_THRESHOLD = 0.20  # 20% mínimo para considerar válida la detección
            
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                return None, confidence
            
            return category, confidence
            
    except Exception:
        return None, 0.0

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

def get_catalog_embeddings():
    """Obtener el diccionario de embeddings del catálogo"""
    return catalog_embeddings