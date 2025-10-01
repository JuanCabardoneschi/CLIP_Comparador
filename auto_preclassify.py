#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 PRE-CLASIFICADOR AUTOMÁTICO GOODY
====================================
Utiliza los embeddings existentes para hacer una pre-clasificación automática
basada en las nuevas 12 categorías. Después se puede verificar y corregir manualmente.

Nuevas Categorías:
- DELANTAL
- AMBO VESTIR HOMBRE – DAMA
- CAMISAS HOMBRE- DAMA
- CASACAS
- ZUECOS
- GORROS – GORRAS
- CARDIGAN HOMBRE – DAMA
- BUZOS
- ZAPATO DAMA
- CHALECO DAMA- HOMBRE
- CHAQUETAS
- REMERAS

Fecha: 30 Septiembre 2025
"""

import json
import os
import numpy as np
from pathlib import Path

# Configuración
EMBEDDINGS_FILE = "catalogo/embeddings.json"
CLASSIFICATIONS_FILE = "catalogo/product_classifications.json"

# Nuevas categorías
CATEGORIES = [
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

# Palabras clave para cada categoría (basadas en los nombres de archivos)
CATEGORY_KEYWORDS = {
    "DELANTAL": [
        "delantal", "delanta", "pechera", "jumper", "alpacuna", "blue note",
        "inesita", "faldon", "western", "zarpado", "bow", "medio delantal"
    ],
    "AMBO VESTIR HOMBRE – DAMA": [
        "ambo", "vestir"
    ],
    "CAMISAS HOMBRE- DAMA": [
        "camisa", "camisas", "boton", "havanna", "hotelga"
    ],
    "CASACAS": [
        "casaca", "mercure", "slim"
    ],
    "ZUECOS": [
        "zueco", "soft works"
    ],
    "GORROS – GORRAS": [
        "gorro", "gorra", "boina", "baccio", "cup", "calada"
    ],
    "CARDIGAN HOMBRE – DAMA": [
        "cardigan"
    ],
    "BUZOS": [
        "buzo", "frizado"
    ],
    "ZAPATO DAMA": [
        "zapato", "piccadilly", "taco"
    ],
    "CHALECO DAMA- HOMBRE": [
        "chaleco"
    ],
    "CHAQUETAS": [
        "chaqueta", "chef", "cuello", "redondo", "skull", "pop", "gin", "punto", "caramelo", "dry-fit", "combinada"
    ],
    "REMERAS": [
        "remera", "playera", "camiseta"  # Agregar si aparecen en el futuro
    ]
}

def load_embeddings():
    """Cargar embeddings existentes"""
    try:
        with open(EMBEDDINGS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo {EMBEDDINGS_FILE}")
        return {}

def classify_by_filename(filename):
    """Clasificar basándose en palabras clave del nombre del archivo"""
    filename_lower = filename.lower()
    
    # Crear scoring por categoría
    category_scores = {}
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in filename_lower:
                # Dar más peso a palabras exactas
                if keyword.lower() == filename_lower.split('.')[0]:
                    score += 10
                else:
                    score += 1
                matched_keywords.append(keyword)
        
        if score > 0:
            category_scores[category] = {
                'score': score,
                'keywords': matched_keywords
            }
    
    # Encontrar la mejor categoría
    if category_scores:
        best_category = max(category_scores.keys(), key=lambda x: category_scores[x]['score'])
        confidence = min(0.9, category_scores[best_category]['score'] / 10)  # Máximo 0.9
        
        return {
            'category': best_category,
            'confidence': confidence,
            'matched_keywords': category_scores[best_category]['keywords'],
            'all_scores': category_scores
        }
    
    # Si no hay coincidencias, clasificar como la categoría más genérica
    return {
        'category': 'DELANTAL',  # Categoría por defecto
        'confidence': 0.1,
        'matched_keywords': [],
        'all_scores': {}
    }

def create_top3_categories(classification_result, all_categories):
    """Crear top 3 categorías basadas en el resultado de clasificación"""
    top_3 = []
    
    # 1. Categoría principal
    top_3.append({
        'category': classification_result['category'],
        'confidence': classification_result['confidence']
    })
    
    # 2. Categorías adicionales basadas en scores
    all_scores = classification_result['all_scores']
    other_categories = [(cat, data['score']) for cat, data in all_scores.items() 
                       if cat != classification_result['category']]
    
    # Ordenar por score y tomar las mejores
    other_categories.sort(key=lambda x: x[1], reverse=True)
    
    for cat, score in other_categories[:2]:  # Tomar las 2 mejores adicionales
        confidence = min(0.8, score / 12)  # Confidence menor que la principal
        top_3.append({
            'category': cat,
            'confidence': confidence
        })
    
    # Completar con categorías aleatorias si es necesario
    while len(top_3) < 3:
        remaining_cats = [cat for cat in all_categories if cat not in [item['category'] for item in top_3]]
        if remaining_cats:
            top_3.append({
                'category': remaining_cats[0],
                'confidence': 0.05
            })
        else:
            break
    
    return top_3

def auto_classify_products():
    """Realizar pre-clasificación automática de todos los productos"""
    print("🤖 INICIANDO PRE-CLASIFICACIÓN AUTOMÁTICA")
    print("=" * 60)
    
    # Cargar embeddings
    embeddings = load_embeddings()
    if not embeddings:
        print("❌ No se pudieron cargar los embeddings")
        return
    
    print(f"📊 Embeddings cargados: {len(embeddings)}")
    print(f"🏷️ Categorías disponibles: {len(CATEGORIES)}")
    print()
    
    # Procesar cada imagen
    classifications = {}
    category_counts = {cat: 0 for cat in CATEGORIES}
    
    for file_path, embedding in embeddings.items():
        # Extraer nombre del archivo
        filename = os.path.basename(file_path)
        
        # Clasificar automáticamente
        result = classify_by_filename(filename)
        
        # Crear top 3 categorías
        top_3 = create_top3_categories(result, CATEGORIES)
        
        # Guardar clasificación
        classifications[filename] = {
            'category': result['category'],
            'confidence': result['confidence'],
            'top_3_categories': top_3
        }
        
        # Actualizar contadores
        category_counts[result['category']] += 1
        
        # Mostrar progreso
        print(f"📷 {filename}")
        print(f"   🏷️ Categoría: {result['category']} (conf: {result['confidence']:.3f})")
        if result['matched_keywords']:
            print(f"   🔍 Palabras clave: {', '.join(result['matched_keywords'])}")
        print()
    
    # Guardar clasificaciones
    with open(CLASSIFICATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(classifications, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("✅ PRE-CLASIFICACIÓN COMPLETADA")
    print(f"📄 Guardado en: {CLASSIFICATIONS_FILE}")
    print()
    
    # Mostrar estadísticas
    print("📊 ESTADÍSTICAS POR CATEGORÍA:")
    print("-" * 40)
    for category, count in category_counts.items():
        if count > 0:
            print(f"🏷️ {category}: {count} productos")
    
    print()
    print("🎯 PRÓXIMOS PASOS:")
    print("1. Ejecutar: python manual_classifier.py")
    print("2. Revisar y corregir clasificaciones en http://localhost:5000")
    print("3. Usar filtros para revisar cada categoría")
    print("4. Eliminar duplicados con el botón 🗑️")
    print("=" * 60)

if __name__ == "__main__":
    auto_classify_products()