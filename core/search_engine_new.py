"""
Motor de búsqueda inteligente
"""

import os
import json
from core.classification import calculate_similarity, get_catalog_embeddings

def find_similar_images(query_embedding, top_k=3, query_type=None, query_confidence=0.0):
    """
    Buscar imágenes similares en el catálogo
    """
    
    catalog_embeddings = get_catalog_embeddings()
    if not catalog_embeddings or query_embedding is None:
        print(f"Error: catalog_embeddings={len(catalog_embeddings) if catalog_embeddings else 0}, query_embedding={query_embedding is not None}")
        return []

    print(f"Catálogo tiene {len(catalog_embeddings)} embeddings")
    print(f"Categoría detectada: '{query_type}' con confianza: {query_confidence}")
    
    # VERIFICAR SI LA CATEGORÍA ES COMERCIALIZADA POR GOODY
    categorias_goody = [
        "buzo", "hoodie", "sudadera", "frizado",
        "camisa", "shirt", "blusa",
        "gorro", "gorra", "cap", "hat", "boina",
        "chaqueta", "jacket", "campera", "casaca",
        "delantal", "pechera", "mandil",
        "chaleco", "cardigan", "vest",
        "ambo", "uniforme médico",
        "zapato", "calzado", "zueco",
        "remera", "polo", "camiseta"
    ]
    
    # Verificar si la categoría detectada está en nuestro catálogo
    categoria_encontrada = False
    if query_type:
        query_lower = query_type.lower()
        for categoria in categorias_goody:
            if categoria in query_lower:
                categoria_encontrada = True
                break
    
    # Si la categoría no está en nuestro catálogo, retornar vacío
    if not categoria_encontrada:
        print(f"❌ Categoría '{query_type}' no comercializada por GOODY")
        return []
    
    print(f"✅ Categoría '{query_type}' es comercializada por GOODY")
    
    # Calcular similitudes visuales
    similarities = []
    for filename, catalog_embedding in catalog_embeddings.items():
        try:
            visual_similarity = calculate_similarity(query_embedding, catalog_embedding)
            similarities.append((filename, visual_similarity))
        except Exception as e:
            print(f"Error calculando similitud con {filename}: {e}")
            continue

    print(f"Calculadas {len(similarities)} similitudes")
    
    # Ordenar por similitud (mayor a menor)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Filtrar por similitud mínima (60%)
    MIN_SIMILARITY = 0.60
    filtered_similarities = [(f, s) for f, s in similarities if s >= MIN_SIMILARITY]
    
    print(f"Después del filtro de similitud mínima ({MIN_SIMILARITY}): {len(filtered_similarities)} resultados")
    
    # Si no hay resultados con alta similitud, ser más permisivo
    if not filtered_similarities:
        MIN_SIMILARITY = 0.40  # Bajar el umbral
        filtered_similarities = [(f, s) for f, s in similarities if s >= MIN_SIMILARITY]
        print(f"Con umbral reducido ({MIN_SIMILARITY}): {len(filtered_similarities)} resultados")
    
    # Tomar los top_k resultados
    results = filtered_similarities[:top_k]
    
    print(f"Top {top_k} resultados finales: {[(f, round(s, 3)) for f, s in results]}")
    
    return results