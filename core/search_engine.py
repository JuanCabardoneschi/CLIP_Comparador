"""
Motor de búsqueda inteligente
"""

import os
import json
from core.classification import calculate_similarity, get_catalog_embeddings

def find_similar_images(query_embedding, top_k=3, query_type=None, query_confidence=0.0):
    """
    Buscar imágenes similares en el catálogo
    Retorna: lista de tuplas (filename, score) o cadena especial para categorías no comercializadas
    """
    
    catalog_embeddings = get_catalog_embeddings()
    if not catalog_embeddings or query_embedding is None:
        print(f"Error: catalog_embeddings={len(catalog_embeddings) if catalog_embeddings else 0}, query_embedding={query_embedding is not None}")
        return []

    print(f"Catálogo tiene {len(catalog_embeddings)} embeddings")
    print(f"Categoría detectada: '{query_type}' con confianza: {query_confidence}")
    
    # Importar configuración centralizada de categorías
    from config.categories import is_commercial_category
    
    # Verificar si la categoría detectada está en nuestro catálogo comercializado
    if query_type and not is_commercial_category(query_type):
        print(f"❌ Categoría '{query_type}' no comercializada por GOODY")
        return []

    print(f"✅ Categoría '{query_type}' es comercializada por GOODY")
    
    # Calcular similitudes visuales
    similarities = []
    category_matches = []  # Para productos de la misma categoría
    
    # Determinar palabras clave de la categoría detectada
    category_keywords = []
    if query_type:
        query_lower = query_type.lower()
        print(f"🔍 Analizando categoría detectada: '{query_lower}'")
        
        # Mapear categorías a palabras clave de archivos
        if any(word in query_lower for word in ['gorro', 'gorra', 'boina', 'cap', 'hat']):
            category_keywords = ['gorro', 'gorra', 'boina']
            print(f"✅ Categoría GORRO/GORRA detectada. Palabras clave: {category_keywords}")
        elif any(word in query_lower for word in ['delantal', 'pechera', 'mandil', 'apron']):
            category_keywords = ['delantal', 'pechera']
            print(f"✅ Categoría DELANTAL detectada. Palabras clave: {category_keywords}")
        elif any(word in query_lower for word in ['camisa', 'shirt', 'blusa']):
            category_keywords = ['camisa']
            print(f"✅ Categoría CAMISA detectada. Palabras clave: {category_keywords}")
        elif any(word in query_lower for word in ['casaca', 'chef']):
            category_keywords = ['casaca']
            print(f"✅ Categoría CASACA detectada. Palabras clave: {category_keywords}")
        elif any(word in query_lower for word in ['chaqueta', 'jacket', 'campera']):
            category_keywords = ['chaqueta']
            print(f"✅ Categoría CHAQUETA detectada. Palabras clave: {category_keywords}")
        elif any(word in query_lower for word in ['buzo', 'hoodie', 'sudadera']):
            category_keywords = ['buzo']
            print(f"✅ Categoría BUZO detectada. Palabras clave: {category_keywords}")
        elif any(word in query_lower for word in ['chaleco', 'cardigan', 'vest']):
            category_keywords = ['chaleco', 'cardigan']
            print(f"✅ Categoría CHALECO/CARDIGAN detectada. Palabras clave: {category_keywords}")
        elif any(word in query_lower for word in ['ambo', 'uniforme', 'scrubs']):
            category_keywords = ['ambo']
            print(f"✅ Categoría AMBO detectada. Palabras clave: {category_keywords}")
        elif any(word in query_lower for word in ['zapato', 'zueco', 'calzado']):
            category_keywords = ['zapato', 'zueco']
            print(f"✅ Categoría ZAPATO/ZUECO detectada. Palabras clave: {category_keywords}")
        elif any(word in query_lower for word in ['remera', 'polo', 'camiseta']):
            category_keywords = ['remera', 'polo']
            print(f"✅ Categoría REMERA/POLO detectada. Palabras clave: {category_keywords}")
        else:
            print(f"❌ Categoría NO MAPEADA: '{query_lower}'")
        
        print(f"🎯 Palabras clave finales para filtrado: {category_keywords}")

    for filename, catalog_embedding in catalog_embeddings.items():
        try:
            visual_similarity = calculate_similarity(query_embedding, catalog_embedding)
            
            # Verificar si es de la misma categoría
            is_same_category = False
            if category_keywords:
                filename_lower = filename.lower()
                is_same_category = any(keyword in filename_lower for keyword in category_keywords)
                
                # Log detallado para debugging
                if is_same_category:
                    print(f"  ✅ MATCH: {filename} (contiene palabras de la categoría)")
                elif any(keyword in filename.lower() for keyword in ['gorro', 'gorra', 'boina']):
                    print(f"  🧢 GORRA: {filename}: {visual_similarity:.3f} {'✅ MATCH' if is_same_category else '❌ NO MATCH'}")
            
            # Aplicar boost de categoría (aumentar similitud en 0.1 para misma categoría)
            boosted_similarity = visual_similarity + (0.1 if is_same_category else 0.0)
            
            similarities.append((filename, visual_similarity, boosted_similarity, is_same_category))
                
        except Exception as e:
            print(f"Error calculando similitud con {filename}: {e}")
            continue

    print(f"Calculadas {len(similarities)} similitudes")
    
    # **FILTRO ABSOLUTO POR CATEGORÍA PRIMERO**
    # Separar productos por categoría
    same_category_products = [(f, s) for f, s, boosted_s, is_cat in similarities if is_cat]
    other_products = [(f, s) for f, s, boosted_s, is_cat in similarities if not is_cat]
    
    print(f"🎯 Productos de la categoría '{query_type}': {len(same_category_products)}")
    print(f"📦 Otros productos: {len(other_products)}")
    
    # **REGLA ABSOLUTA: Si hay productos de la misma categoría, SOLO usar esos**
    if same_category_products:
        print(f"✅ FILTRO ABSOLUTO: Solo mostrando productos de la categoría '{query_type}'")
        
        # Ordenar productos de la misma categoría por similitud
        same_category_products.sort(key=lambda x: x[1], reverse=True)
        
        # Aplicar umbral mínimo más permisivo para la misma categoría
        MIN_CATEGORY_SIMILARITY = 0.30  # Umbral más bajo para misma categoría
        filtered_similarities = [(f, s) for f, s in same_category_products if s >= MIN_CATEGORY_SIMILARITY]
        
        print(f"🔍 Productos de la categoría con similitud >= {MIN_CATEGORY_SIMILARITY}: {len(filtered_similarities)}")
        
        # Si aún no hay suficientes, tomar todos los de la categoría
        if not filtered_similarities:
            filtered_similarities = same_category_products
            print(f"⚠️  Tomando todos los productos de la categoría sin filtro de similitud")
            
    else:
        # Solo si NO hay productos de la misma categoría, usar algoritmo estándar
        print("📊 No hay productos de la misma categoría, usando algoritmo estándar")
        original_similarities = [(f, s) for f, s, boosted_s, is_cat in similarities]
        original_similarities.sort(key=lambda x: x[1], reverse=True)
        
        filtered_similarities = [(f, s) for f, s in original_similarities if s >= 0.60]
        if not filtered_similarities:
            filtered_similarities = [(f, s) for f, s in original_similarities if s >= 0.40]
    
    print(f"🎯 Resultados después del filtro absoluto por categoría: {len(filtered_similarities)}")
    
    # Tomar los top_k resultados
    results = filtered_similarities[:top_k]
    
    print(f"Top {top_k} resultados finales: {[(f, round(s, 3)) for f, s in results]}")
    
    return results