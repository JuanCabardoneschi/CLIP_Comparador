# 🔍 CLIP Comparador - Sistema de Búsqueda Visual Inteligente# CLIP Comparador - Interfaz Web



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)## 🎯 Descripción

[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)Interfaz web simple para subir imágenes y encontrar prendas similares usando el modelo CLIP de OpenAI.

[![CLIP](https://img.shields.io/badge/OpenAI-CLIP-orange.svg)](https://github.com/openai/CLIP)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)## 🚀 Cómo usar



Sistema web inteligente para encontrar productos similares usando búsqueda visual con el modelo CLIP de OpenAI. Desarrollado específicamente para catálogos de ropa profesional y uniformes.### 1. Preparar el catálogo

Coloca imágenes de prendas en la carpeta `catalogo/`:

## 🎯 **Características Principales**- Formatos soportados: PNG, JPG, JPEG, GIF, BMP, WEBP

- Tamaño máximo: 16MB por imagen

### ✅ **Implementado**

- 🔍 **Búsqueda Visual Inteligente** - Usando modelo CLIP ViT-B/32### 2. Ejecutar la aplicación

- 📱 **Interfaz Web Responsive** - Drag & drop para subir imágenes```bash

- 🎯 **Detección Automática** - Clasifica automáticamente el tipo de prendapython main.py

- 🔄 **Búsqueda Expandida** - Encuentra alternativas visuales (ej: camisas para remeras)```

- ⚡ **Cache Optimizado** - Embeddings pre-calculados para velocidad

- 🚫 **Filtrado Inteligente** - Detecta productos no comercializados### 3. Abrir en navegador

- 📊 **Top-3 Resultados** - Muestra los productos más similaresVisita: http://localhost:5000

- 💾 **Procesamiento en Memoria** - Sin archivos temporales

## 📋 Características

### 🔄 **Próximas Mejoras**

- [ ] Base de datos persistente### ✅ Implementado

- [ ] API REST completa- ✅ Interfaz web responsive

- [ ] Filtros por categoría- ✅ Carga del modelo CLIP ViT-B/32

- [ ] Búsqueda por texto- ✅ Procesamiento de imágenes del catálogo

- [ ] Metadatos de productos (precio, stock)- ✅ Subida de imágenes por drag & drop

- [ ] Sistema de usuarios- ✅ Generación de embeddings CLIP

- ✅ Comparación por similitud coseno

## 🏷️ **Categorías Soportadas**- ✅ Visualización de Top-5 resultados

- ✅ Cache de embeddings del catálogo

El sistema detecta y busca en 12 categorías principales:- ✅ Estado del sistema en tiempo real



| Categoría | Productos | Estado |### 🔄 Próximas mejoras

|-----------|-----------|---------|- [ ] Metadatos de productos (precio, categoría)

| 🥼 **DELANTAL** | Delantales con pechera, mandiles | ✅ 24 productos |- [ ] Filtros por categoría

| 👔 **CAMISAS** | Camisas profesionales, formales | ✅ 8 productos |- [ ] Búsqueda por texto

| 🧥 **CHAQUETAS** | Chaquetas chef, profesionales | ✅ Disponible |- [ ] Base de datos persistente

| 🦺 **CASACAS** | Casacas médicas, laboratorio | ✅ Disponible |- [ ] API REST completa

| 👥 **AMBO** | Uniformes médicos completos | ✅ Disponible |

| 🧶 **CARDIGAN** | Cardigans, chalecos tejidos | ✅ Disponible |## 🛠️ Estructura del proyecto

| 🧢 **GORROS** | Gorros chef, profesionales | ✅ Disponible |```

| 👕 **BUZOS** | Sudaderas, buzos | ✅ Disponible |CLIP_Comparador/

| 🦺 **CHALECO** | Chalecos formales | ✅ Disponible |├── main.py              # Aplicación Flask principal

| 👟 **CALZADO** | Zapatos, zuecos profesionales | ✅ Disponible |├── templates/

| 👕 **REMERAS** | Polos, remeras (→ busca en camisas) | ⚡ Búsqueda expandida |│   └── index.html       # Interfaz web

| ❌ **Otros** | Pantalones, faldas, vestidos | 🚫 No comercializado |├── static/              # Archivos estáticos (CSS, JS)

├── uploads/             # Imágenes subidas por usuarios

## 🚀 **Instalación y Uso**├── catalogo/            # Catálogo de prendas

│   └── embeddings.json  # Cache de embeddings

### **Prerrequisitos**└── venv/               # Entorno virtual Python

- Python 3.8 o superior```

- 4GB+ RAM (para modelo CLIP)

- Espacio en disco: ~2GB## 📊 API Endpoints



### **1. Clonar el repositorio**### GET /

```bashPágina principal de la interfaz

git clone https://github.com/tu-usuario/CLIP_Comparador.git

cd CLIP_Comparador### POST /upload

```Subir imagen y encontrar similares

- Input: Archivo de imagen

### **2. Crear entorno virtual**- Output: JSON con imágenes similares y porcentajes

```bash

python -m venv venv### GET /status

Estado del sistema

# Windows- Output: Estado del modelo, catálogo y dispositivo

venv\Scripts\activate

## 💡 Consejos de uso

# Linux/Mac

source venv/bin/activate1. **Calidad de imágenes**: Usa imágenes claras y bien iluminadas

```2. **Catálogo diverso**: Incluye variedad de prendas para mejores resultados

3. **Rendimiento**: Los embeddings se calculan una vez y se guardan en cache

### **3. Instalar dependencias**4. **CPU vs GPU**: El sistema detecta automáticamente si hay GPU disponible
```bash
pip install -r requirements.txt
```

### **4. Preparar el catálogo**
1. Coloca las imágenes de productos en la carpeta `catalogo/`
2. Formatos soportados: PNG, JPG, JPEG, GIF, BMP, WEBP
3. Tamaño máximo: 16MB por imagen

### **5. Generar embeddings del catálogo**
```bash
python generate_embeddings.py
```

### **6. Ejecutar la aplicación**
```bash
python app_simple.py
```

### **7. Abrir en navegador**
Visita: **http://localhost:5000**

## 🛠️ **Estructura del Proyecto**

```
CLIP_Comparador/
├── app_simple.py              # 🚀 Aplicación Flask principal
├── generate_embeddings.py     # 🔧 Generador de embeddings CLIP
├── manual_classifier.py       # 🏷️ Clasificador manual de productos  
├── auto_preclassify.py        # 🤖 Pre-clasificador automático
├── requirements.txt           # 📦 Dependencias Python
├── README.md                  # 📖 Documentación
├── .gitignore                 # 🚫 Archivos ignorados por Git
├── templates/                 # 🎨 Plantillas HTML
│   ├── index.html            # 🏠 Interfaz principal
│   └── manual_classifier.html # ⚙️ Herramienta de clasificación
├── static/                    # 📁 Archivos estáticos (CSS, JS)
├── uploads/                   # 📤 Imágenes subidas (temporal)
├── catalogo/                  # 🗃️ Catálogo de productos
│   ├── embeddings.json       # 💾 Cache de embeddings
│   ├── product_classifications.json # 🏷️ Clasificaciones
│   └── *.jpg                 # 🖼️ Imágenes del catálogo
└── venv/                      # 🐍 Entorno virtual Python
```

## 🔧 **Tecnologías Utilizadas**

- **Backend:** Python 3.10, Flask 2.3
- **IA/ML:** OpenAI CLIP, PyTorch, NumPy
- **Frontend:** HTML5, CSS3, JavaScript
- **Procesamiento:** Pillow (PIL), Base64
- **Cache:** JSON embeddings

## 📊 **Algoritmo de Búsqueda**

### **Proceso de Búsqueda Inteligente v3.8.1:**

1. **📤 Subida de Imagen** - Usuario sube imagen (drag & drop)
2. **🔍 Generación de Embedding** - CLIP procesa la imagen
3. **🤖 Clasificación Automática** - Detecta tipo de prenda
4. **⚡ Búsqueda Expandida** - Si no hay stock, busca alternativas
5. **📏 Cálculo de Similitud** - Similitud coseno vs catálogo
6. **🎯 Filtrado Inteligente** - Umbral mínimo 60% similitud
7. **🏆 Top-3 Resultados** - Muestra los más similares

### **Innovación: Búsqueda Expandida**
```python
# Ejemplo: Remera Polo → Camisas
if no_hay_remeras_en_stock:
    buscar_en_camisas_similares()  # Expansión inteligente
```

## 📈 **Métricas de Rendimiento**

- **Precisión:** >85% en detección de categorías
- **Velocidad:** <2s respuesta (con cache)
- **Catálogo:** 57+ productos indexados
- **Similitud mínima:** 60% para resultados relevantes

## 🤝 **Contribuir**

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 **Changelog**

### **v3.8.1** (2025-09-30)
- ✅ **Corrección crítica:** Búsqueda expandida para remeras → camisas
- ✅ **Fix:** Imagen subida se muestra correctamente (base64)
- ✅ **Mejora:** Termina búsqueda al detectar categorías no comercializadas

### **v3.8.0** (2025-09-29)  
- ✅ **Detección ampliada:** Categorías no comercializadas
- ✅ **Umbral de similitud:** Rechaza resultados <60%
- ✅ **Enfoque simplificado:** Verificación genérica de categorías

## 📄 **Licencia**

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 **Autores**

- **Tu Nombre** - *Desarrollo inicial* - [@tu-usuario](https://github.com/tu-usuario)

## 🙏 **Agradecimientos**

- [OpenAI CLIP](https://github.com/openai/CLIP) - Modelo de búsqueda visual
- [Flask](https://flask.palletsprojects.com/) - Framework web
- [PyTorch](https://pytorch.org/) - Framework de machine learning

---

⭐ **¡Dale una estrella al proyecto si te resulta útil!**