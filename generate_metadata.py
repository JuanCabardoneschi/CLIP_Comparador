#!/usr/bin/env python3
"""
Generador Automático de Metadata para CLIP Comparador
Genera metadata inicial basada en nombres de archivos y análisis CLIP
"""

import os
import json
import random
import re
from datetime import datetime, timedelta
import numpy as np

# Importar funciones del sistema principal
import sys
sys.path.append('.')

def setup_clip():
    """Configurar CLIP para análisis de colores"""
    try:
        import torch
        import clip
        from PIL import Image
        
        device = "cpu"
        model, preprocess = clip.load("RN50", device=device)
        model.eval()
        
        return model, preprocess, device
    except Exception as e:
        print(f"Error configurando CLIP: {e}")
        return None, None, None

def extract_category_from_filename(filename):
    """Extraer categoría del nombre del archivo"""
    filename_lower = filename.lower()
    
    # Mapeo de palabras clave a categorías GOODY
    category_keywords = {
        'delantal': 'delantal',
        'ambo': 'ambo',
        'camisa': 'camisa',
        'casaca': 'casaca',
        'zueco': 'zueco',
        'gorro': 'gorro',
        'boina': 'gorro',
        'cardigan': 'cardigan',
        'buzo': 'buzo',
        'zapato': 'zapato dama',
        'chaleco': 'chaleco',
        'chaqueta': 'chaqueta'
    }
    
    # Buscar palabras clave en el nombre del archivo
    for keyword, category in category_keywords.items():
        if keyword in filename_lower:
            return category
    
    # Si no encuentra ninguna categoría específica, intentar inferir
    if 'dama' in filename_lower or 'mujer' in filename_lower:
        return 'ambo'  # Default para ropa de dama
    elif 'hombre' in filename_lower:
        return 'ambo'  # Default para ropa de hombre
    
    return 'camisa'  # Default general

def extract_product_name_from_filename(filename):
    """Extraer nombre del producto desde el nombre del archivo"""
    # Remover extensión
    name = os.path.splitext(filename)[0]
    
    # Limpiar caracteres especiales y números al final
    name = re.sub(r'\s*\(\d+\)\.?\w*$', '', name)  # Remover (1), (2), etc.
    name = re.sub(r'\s*\d+\.?\w*$', '', name)      # Remover números al final
    
    # Capitalizar palabras importantes
    words = name.split()
    cleaned_words = []
    
    for word in words:
        # Mantener palabras importantes en mayúscula
        if word.upper() in ['NEW', 'SLIM', 'DRY-FIT', 'MOD', 'MODELO']:
            cleaned_words.append(word.upper())
        elif len(word) > 2:
            cleaned_words.append(word.capitalize())
        else:
            cleaned_words.append(word.lower())
    
    return ' '.join(cleaned_words)

def analyze_color_from_image(model, preprocess, device, image_path):
    """Analizar color predominante usando CLIP"""
    if not model:
        return ""
    
    try:
        from PIL import Image
        import torch
        import clip
        
        # Cargar imagen
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Lista de colores para analizar
        color_prompts = [
            "a photo of black clothing",
            "a photo of white clothing", 
            "a photo of blue clothing",
            "a photo of navy blue clothing",
            "a photo of red clothing",
            "a photo of green clothing",
            "a photo of gray clothing",
            "a photo of brown clothing",
            "a photo of beige clothing",
            "a photo of yellow clothing",
            "a photo of pink clothing",
            "a photo of purple clothing"
        ]
        
        color_names = [
            "Negro", "Blanco", "Azul", "Azul Marino", "Rojo", 
            "Verde", "Gris", "Marrón", "Beige", "Amarillo", 
            "Rosa", "Púrpura"
        ]
        
        # Tokenizar colores
        text_tokens = clip.tokenize(color_prompts).to(device)
        
        with torch.no_grad():
            # Obtener embeddings
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            
            # Normalizar
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calcular similitudes
            similarities = (image_features @ text_features.T).squeeze(0)
            
            # Obtener el color más probable
            best_color_idx = similarities.argmax().item()
            confidence = float(similarities[best_color_idx].item())
            
            # Solo retornar si la confianza es razonable
            if confidence > 0.2:
                return color_names[best_color_idx]
        
        return ""
        
    except Exception as e:
        print(f"Error analizando color para {image_path}: {e}")
        return ""

def infer_material_from_category_and_name(category, product_name):
    """Inferir material basado en categoría y nombre del producto"""
    name_lower = product_name.lower()
    
    material_keywords = {
        'frizado': '100% Algodón Frizado',
        'dry-fit': 'Poliéster Dry-Fit',
        'jean': 'Denim',
        'alpacuna': '60% Algodón, 40% Poliéster',
        'polo': '100% Algodón Piqué',
        'gabardina': 'Gabardina',
        'lycra': '95% Algodón, 5% Lycra'
    }
    
    # Buscar palabras clave en el nombre
    for keyword, material in material_keywords.items():
        if keyword in name_lower:
            return material
    
    # Material por categoría
    category_materials = {
        'delantal': '100% Algodón',
        'ambo': '65% Poliéster, 35% Algodón',
        'camisa': '100% Algodón',
        'casaca': '65% Poliéster, 35% Algodón',
        'zueco': 'EVA',
        'gorro': '100% Algodón',
        'cardigan': '100% Algodón',
        'buzo': '80% Algodón, 20% Poliéster',
        'zapato dama': 'Cuero sintético',
        'chaleco': '65% Poliéster, 35% Algodón',
        'chaqueta': '100% Algodón'
    }
    
    return category_materials.get(category, '100% Algodón')

def generate_random_sizes():
    """Generar 3 talles aleatorios"""
    all_sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', 'Único']
    
    # Para productos únicos (gorros, zuecos), usar talle único
    unique_categories = ['gorro', 'zueco', 'zapato dama']
    
    # Probabilidades para diferentes combinaciones
    common_combinations = [
        ['S', 'M', 'L'],
        ['M', 'L', 'XL'],
        ['S', 'M', 'XL'],
        ['XS', 'S', 'M'],
        ['L', 'XL', 'XXL'],
        ['M', 'L', 'XXL'],
        ['S', 'L', 'XL']
    ]
    
    return random.choice(common_combinations)

def generate_random_date():
    """Generar fecha aleatoria de los últimos 6 meses"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 meses atrás
    
    # Generar fecha aleatoria
    random_days = random.randint(0, 180)
    random_date = start_date + timedelta(days=random_days)
    
    return random_date.strftime('%Y-%m-%d')

def get_catalog_images():
    """Obtener lista de imágenes del catálogo"""
    catalog_path = 'catalogo'
    images = []
    
    if os.path.exists(catalog_path):
        for filename in os.listdir(catalog_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images.append(filename)
    
    return sorted(images)

def generate_metadata_for_all_images():
    """Generar metadata para todas las imágenes del catálogo"""
    print("🔄 Iniciando generación automática de metadata...")
    
    # Configurar CLIP para análisis de colores
    print("📦 Configurando CLIP para análisis de colores...")
    model, preprocess, device = setup_clip()
    
    # Obtener imágenes del catálogo
    catalog_images = get_catalog_images()
    print(f"📁 Encontradas {len(catalog_images)} imágenes en el catálogo")
    
    # Cargar metadata existente (si existe)
    metadata_file = "metadata.json"
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            existing_metadata = json.load(f)
        print(f"📋 Cargada metadata existente para {len(existing_metadata)} productos")
    except FileNotFoundError:
        existing_metadata = {}
        print("📋 No se encontró metadata existente, creando desde cero")
    
    generated_metadata = {}
    
    for i, filename in enumerate(catalog_images, 1):
        print(f"\n🔍 Procesando {i}/{len(catalog_images)}: {filename}")
        
        # Si ya existe metadata, conservarla
        if filename in existing_metadata:
            generated_metadata[filename] = existing_metadata[filename]
            print(f"   ✅ Metadata existente conservada")
            continue
        
        try:
            # 1. Extraer categoría del nombre
            category = extract_category_from_filename(filename)
            print(f"   📂 Categoría detectada: {category}")
            
            # 2. Extraer nombre del producto
            product_name = extract_product_name_from_filename(filename)
            print(f"   🏷️ Nombre del producto: {product_name}")
            
            # 3. Analizar color predominante
            image_path = os.path.join('catalogo', filename)
            color = analyze_color_from_image(model, preprocess, device, image_path)
            print(f"   🎨 Color detectado: {color if color else 'No detectado'}")
            
            # 4. Inferir material
            material = infer_material_from_category_and_name(category, product_name)
            print(f"   🧵 Material inferido: {material}")
            
            # 5. Generar talles aleatorios
            if category in ['gorro', 'zueco', 'zapato dama']:
                sizes = ['Único']
            else:
                sizes = generate_random_sizes()
            print(f"   📏 Talles generados: {sizes}")
            
            # 6. Generar datos aleatorios
            stock = random.randint(10, 300)
            precio = random.randint(5000, 25000)
            fecha_ingreso = generate_random_date()
            
            print(f"   💰 Precio: ${precio:,}")
            print(f"   📦 Stock: {stock}")
            print(f"   📅 Fecha ingreso: {fecha_ingreso}")
            
            # 7. Crear metadata completa
            metadata = {
                'nombre': product_name,
                'categoria': category,
                'tipo_prenda': product_name,  # Usar el mismo nombre como tipo
                'color': color,
                'material': material,
                'talle': sizes,
                'precio': float(precio),
                'stock': stock,
                'fecha_ingreso': fecha_ingreso,
                'notas': f'Metadata generada automáticamente desde {filename}'
            }
            
            generated_metadata[filename] = metadata
            print(f"   ✅ Metadata generada exitosamente")
            
        except Exception as e:
            print(f"   ❌ Error procesando {filename}: {e}")
            # Crear metadata básica en caso de error
            generated_metadata[filename] = {
                'nombre': extract_product_name_from_filename(filename),
                'categoria': 'camisa',  # Default
                'tipo_prenda': extract_product_name_from_filename(filename),
                'color': '',
                'material': '100% Algodón',
                'talle': ['M', 'L', 'XL'],
                'precio': 15000.0,
                'stock': 50,
                'fecha_ingreso': generate_random_date(),
                'notas': f'Metadata básica generada por error en procesamiento de {filename}'
            }
    
    # Guardar metadata generada
    print(f"\n💾 Guardando metadata para {len(generated_metadata)} productos...")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(generated_metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ Metadata guardada exitosamente en {metadata_file}")
        
        # Estadísticas finales
        categories = {}
        total_value = 0
        total_stock = 0
        
        for data in generated_metadata.values():
            cat = data['categoria']
            categories[cat] = categories.get(cat, 0) + 1
            total_value += data['precio'] * data['stock']
            total_stock += data['stock']
        
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   📦 Total productos: {len(generated_metadata)}")
        print(f"   📂 Categorías:")
        for cat, count in sorted(categories.items()):
            print(f"      • {cat.title()}: {count} productos")
        print(f"   💰 Valor total inventario: ${total_value:,}")
        print(f"   📦 Stock total: {total_stock:,} unidades")
        
        return True
        
    except Exception as e:
        print(f"❌ Error guardando metadata: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Generador Automático de Metadata - CLIP Comparador")
    print("=" * 60)
    
    success = generate_metadata_for_all_images()
    
    if success:
        print(f"\n🎉 ¡Metadata generada exitosamente!")
        print(f"🔗 Ahora puedes usar el panel de administración para revisar y editar la metadata")
        print(f"🌐 Inicia sesión como 'admin' en http://127.0.0.1:5000")
    else:
        print(f"\n❌ Error generando metadata")
        
    print("\n" + "=" * 60)