# Categorías Oficiales GOODY - Sistema CLIP Comparador

## 📋 **CATEGORÍAS COMERCIALIZADAS** *(12 categorías oficiales)*

### 1. **DELANTAL**
- Palabras clave: `delantal`, `pechera`, `mandil`, `apron`
- Descripción CLIP: *"delantal de trabajo con pechera, mandil profesional con tirantes, apron with straps"*

### 2. **AMBO VESTIR HOMBRE – DAMA**
- Palabras clave: `ambo`, `uniforme médico`, `scrubs`, `medical uniform`
- Descripción CLIP: *"ambo vestir hombre y dama, uniforme médico scrubs, medical uniform set"*

### 3. **CAMISAS HOMBRE- DAMA**
- Palabras clave: `camisa`, `shirt`, `blusa`, `dress shirt`
- Descripción CLIP: *"camisa hombre y dama, blusa formal de trabajo, dress shirt with collar"*

### 4. **CASACAS**
- Palabras clave: `casaca`, `chef jacket`, `chaqueta de cocina`
- Descripción CLIP: *"casaca de chef profesional, chaqueta de cocina blanca, chef jacket with buttons"*

### 5. **ZUECOS**
- Palabras clave: `zueco`, `clogs`, `calzado profesional`
- Descripción CLIP: *"zueco profesional de trabajo, calzado antideslizante, work clogs shoes"*

### 6. **GORROS – GORRAS**
- Palabras clave: `gorro`, `gorra`, `cap`, `hat`, `boina`
- Descripción CLIP: *"gorro de chef y gorra con visera, work cap hat professional"*

### 7. **CARDIGAN HOMBRE – DAMA**
- Palabras clave: `cardigan`, `chaleco con botones`, `sweater`
- Descripción CLIP: *"cardigan hombre y dama, chaleco con botones, cardigan sweater"*

### 8. **BUZOS**
- Palabras clave: `buzo`, `hoodie`, `sudadera`, `sweatshirt`
- Descripción CLIP: *"buzo cerrado con capucha, sudadera de trabajo, hoodie sweatshirt"*

### 9. **ZAPATO DAMA**
- Palabras clave: `zapato dama`, `calzado femenino`, `women shoes`
- Descripción CLIP: *"zapato dama profesional, calzado femenino de trabajo, women work shoes"*

### 10. **CHALECO DAMA- HOMBRE**
- Palabras clave: `chaleco`, `vest`, `sin mangas`
- Descripción CLIP: *"chaleco dama y hombre, vest sin mangas profesional, sleeveless vest"*

### 11. **CHAQUETAS**
- Palabras clave: `chaqueta`, `jacket`, `campera`
- Descripción CLIP: *"chaqueta profesional cerrada, jacket campera de trabajo, work jacket"*

### 12. **REMERAS**
- Palabras clave: `remera`, `polo`, `camiseta`, `t-shirt`
- Descripción CLIP: *"remera casual polo, camiseta sin botones, t-shirt cotton casual"*

---

## 🚫 **CATEGORÍAS NO COMERCIALIZADAS** *(Generan mensaje amigable)*

Estas categorías son detectadas por el sistema pero GOODY no las comercializa:

- **Vestidos:** `vestido`, `dress`
- **Faldas:** `falda`, `skirt`
- **Pantalones:** `pantalón`, `pants`, `jeans`, `trousers`
- **Corbatas:** `corbata`, `tie`
- **Cinturones:** `cinturón`, `belt`
- **Medias:** `medias`, `socks`
- **Ropa interior:** `ropa interior`, `underwear`
- **Shorts:** `short`, `bermuda`, `shorts`

### Mensaje para categorías no comercializadas:
> *"Hemos detectado que buscas '[categoría]'. Actualmente nuestro catálogo se especializa en uniformes profesionales, delantales, camisas de trabajo, chaquetas, buzos, gorros y calzado laboral. ¡Te invitamos a explorar nuestros productos disponibles!"*

---

## ⚙️ **Configuración Técnica**

### Archivos de configuración:
- **Centralizada:** `config/categories.py` - Configuración principal
- **Clasificación:** `core/classification.py` - Usa categorías para CLIP
- **Búsqueda:** `core/search_engine.py` - Filtros de comercialización
- **Rutas:** `routes/main_routes.py` - Validación en endpoints

### Parámetros:
- **Umbral mínimo clasificación:** 20% (0.20)
- **Umbral mínimo similitud:** 60% (0.60) - baja a 40% si no hay resultados
- **Resultados máximos:** Top 3
- **Respuesta HTTP:** Status 200 para ambos casos (comercializadas y no comercializadas)

### Funciones principales:
- `is_commercial_category(text)` - Verifica si es categoría comercializada
- `is_non_commercial_category(text)` - Verifica si es categoría NO comercializada
- `get_clip_categories()` - Obtiene categorías para clasificación CLIP

---

## 🎯 **Flujo del Sistema**

1. **Usuario sube imagen** → Sistema detecta categoría con CLIP
2. **Validación comercial** → Verifica si GOODY comercializa esa categoría
3. **Si es comercializada** → Busca productos similares en catálogo
4. **Si NO es comercializada** → Muestra mensaje amigable de redirección
5. **Frontend** → Presenta resultados o mensaje profesional

**Fecha de actualización:** Octubre 5, 2025
**Versión:** Sistema Modularizado con 12 categorías oficiales GOODY