# Documentación CLIP Comparador

## Capturas de Pantalla

Aquí puedes agregar imágenes demostrativas del sistema en funcionamiento.

### Interfaz Principal
![Interfaz Principal](demo-screenshot.png)

### Resultados de Búsqueda
![Resultados](search-results.png)

## API Endpoints

### GET /
- **Descripción:** Página principal de la interfaz
- **Respuesta:** HTML de la aplicación

### POST /upload
- **Descripción:** Subir imagen y encontrar productos similares
- **Input:** Archivo de imagen (multipart/form-data)
- **Output:** JSON con resultados
```json
{
  "uploaded_file": "imagen.jpg",
  "uploaded_image_data": "data:image/jpeg;base64,/9j/4AAQ...",
  "query_type": "camisa con botones y cuello",
  "query_confidence": 0.275,
  "similar_images": [
    {
      "filename": "CAMISA_BOTON_OCULTO_NEGRA.jpg",
      "similarity": 0.818,
      "similarity_percent": 81.8
    }
  ],
  "status_message": null
}
```

### GET /status
- **Descripción:** Estado del sistema
- **Output:** JSON con información del sistema
```json
{
  "version": "3.8.1",
  "model_loaded": true,
  "catalog_size": 57,
  "device": "cpu"
}
```

## Configuración Avanzada

### Variables de Entorno
- `FLASK_ENV`: Entorno de Flask (development/production)
- `FLASK_DEBUG`: Modo debug (True/False)

### Configuración del Modelo
- Modelo por defecto: `ViT-B/32`
- Dispositivo: Auto-detecta GPU/CPU
- Cache: Embeddings en JSON

### Personalización
- Modificar umbrales en `app_simple.py`
- Ajustar categorías en el código principal
- Personalizar interfaz en `templates/`