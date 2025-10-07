import os
import json
import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path

def get_image_embedding(image_path, model, preprocess, device):
    """Genera embedding combinado de imagen + nombre del archivo"""
    try:
        print(f"🔄 Procesando: {os.path.basename(image_path)}")
        
        # Cargar y procesar imagen
        image = Image.open(image_path).convert('RGB')
        print(f"   📏 Tamaño original: {image.size}")
        
        # Preprocesar imagen
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Procesar nombre del archivo como texto descriptivo
        filename = os.path.basename(image_path)
        # Limpiar nombre del archivo para usarlo como descripción
        text_description = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('(', '').replace(')', '').replace('-', ' ').replace('_', ' ')
        text_description = f"ropa profesional {text_description}"
        
        print(f"   📝 Descripción: {text_description}")
        
        # Generar embeddings
        with torch.no_grad():
            # Embedding de imagen
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Embedding de texto del nombre
            text_tokens = clip.tokenize([text_description]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Combinar embeddings (80% imagen, 20% texto del nombre)
            combined_features = 0.8 * image_features + 0.2 * text_features
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
        
        embedding = combined_features.cpu().numpy().flatten().astype(float)
        print(f"   ✅ Embedding combinado generado - Norma: {np.linalg.norm(embedding):.4f}")
        
        return embedding
        
    except Exception as e:
        print(f"   ❌ Error procesando {image_path}: {str(e)}")
        return None

def generate_catalog_embeddings():
    """Genera embeddings para todo el catálogo"""
    
    print("🚀 Iniciando generación de embeddings v3.9.19...")
    print("🔥 UPGRADE: Modelos ViT-B/16 → ViT-B/32 → RN50 (fallback)")
    
    # Configurar dispositivo
    device = "cpu"  # Forzar CPU para compatibilidad
    print(f"🔧 Dispositivo: {device}")
    
    # Cargar modelo CLIP con fallback inteligente
    models_to_try = [
        ("ViT-B/16", "Mayor precisión visual"),
        ("ViT-B/32", "Buena precisión visual"),  
        ("RN50", "Modelo de respaldo")
    ]
    
    model = None
    preprocess = None
    model_name_used = None
    
    for model_name, description in models_to_try:
        try:
            print(f"🔄 Intentando cargar modelo: {model_name} ({description})")
            model, preprocess = clip.load(model_name, device=device)
            
            if model is not None:
                print(f"✅ Modelo {model_name} cargado exitosamente")
                model_name_used = model_name
                break
                
        except Exception as e:
            print(f"❌ Error cargando {model_name}: {e}")
            continue
    
    if model is None:
        print(f"❌ No se pudo cargar ningún modelo CLIP")
        return False
    
    print(f"🎯 Usando modelo: {model_name_used}")
    model.eval()
    
    # Directorio del catálogo
    catalog_dir = Path("catalogo")
    
    # Buscar todas las imágenes (excluyendo removed_duplicates)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for ext in image_extensions:
        for file in catalog_dir.glob(f"*{ext}"):
            if "removed_duplicates" not in str(file):
                image_files.append(file)
        for file in catalog_dir.glob(f"*{ext.upper()}"):
            if "removed_duplicates" not in str(file):
                image_files.append(file)
    
    print(f"📁 Encontradas {len(image_files)} imágenes")
    
    embeddings = {}
    successful = 0
    failed = 0
    
    # Procesar cada imagen
    for i, image_path in enumerate(image_files, 1):
        print(f"\n📊 Progreso: {i}/{len(image_files)}")
        
        embedding = get_image_embedding(str(image_path), model, preprocess, device)
        
        if embedding is not None:
            # Usar ruta relativa como clave
            rel_path = str(image_path)
            embeddings[rel_path] = embedding.tolist()
            successful += 1
            print(f"   ✅ Guardado: {rel_path}")
        else:
            failed += 1
            print(f"   ❌ Falló: {image_path}")
    
    # Guardar embeddings
    if embeddings:
        embeddings_file = catalog_dir / "embeddings.json"
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings, f, indent=2)
        
        print(f"\n✅ Embeddings guardados: {embeddings_file}")
        print(f"📊 Procesados exitosamente: {successful}")
        print(f"📊 Fallaron: {failed}")
        print(f"📊 Total embeddings: {len(embeddings)}")
    else:
        print("\n❌ No se pudieron generar embeddings")

if __name__ == "__main__":
    generate_catalog_embeddings()