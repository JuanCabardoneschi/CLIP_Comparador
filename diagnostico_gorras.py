"""
Script de diagnóstico para gorras/gorros en el catálogo
"""

import os
import json
import numpy as np

def listar_imagenes_gorras():
    """Listar todas las imágenes de gorras/gorros en el catálogo"""
    catalogo_path = "catalogo"
    
    keywords_gorras = ['gorro', 'gorra', 'boina', 'cap', 'hat']
    
    print("🧢 IMÁGENES DE GORRAS/GORROS EN EL CATÁLOGO:")
    print("=" * 50)
    
    gorras_encontradas = []
    
    if os.path.exists(catalogo_path):
        for filename in os.listdir(catalogo_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename_lower = filename.lower()
                
                # Verificar si contiene palabras clave de gorras
                for keyword in keywords_gorras:
                    if keyword in filename_lower:
                        gorras_encontradas.append(filename)
                        print(f"✅ {filename}")
                        break
    
    print(f"\nTotal encontradas: {len(gorras_encontradas)}")
    return gorras_encontradas

def verificar_embeddings_gorras():
    """Verificar si las gorras tienen embeddings generados"""
    embeddings_file = "catalogo/embeddings.json"
    
    if not os.path.exists(embeddings_file):
        print("❌ No existe archivo embeddings.json")
        return
    
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)
    
    print("\n🧢 VERIFICACIÓN DE EMBEDDINGS PARA GORRAS:")
    print("=" * 50)
    
    keywords_gorras = ['gorro', 'gorra', 'boina', 'cap', 'hat']
    
    gorras_con_embeddings = []
    
    for filename, embedding_list in embeddings_data.items():
        filename_lower = filename.lower()
        
        for keyword in keywords_gorras:
            if keyword in filename_lower:
                gorras_con_embeddings.append(filename)
                embedding_array = np.array(embedding_list)
                print(f"✅ {filename}: embedding shape {embedding_array.shape}")
                break
    
    print(f"\nGorras con embeddings: {len(gorras_con_embeddings)}")

if __name__ == "__main__":
    # Cambiar al directorio del proyecto
    os.chdir("c:/Personal/CLIP_Comparador")
    
    # Ejecutar diagnósticos
    gorras_imagenes = listar_imagenes_gorras()
    verificar_embeddings_gorras()
    
    print("\n📋 RESUMEN:")
    print("=" * 50)
    print("Para mejorar la detección de gorras:")
    print("1. Verificar que todas las gorras tengan embeddings")
    print("2. Revisar los nombres de archivos para consistencia")
    print("3. Ajustar parámetros de similitud si es necesario")
    print("4. Considerar re-generar embeddings si hay inconsistencias")