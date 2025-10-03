# 🔧 Panel de Administración GOODY - Gestión de Metadata

## 📋 Descripción

Panel de control administrativo completo para gestionar la metadata, categorías y configuraciones del sistema CLIP Comparador GOODY.

## 🚀 Características Principales

### 🔐 **Sistema de Autenticación**
- Login administrativo seguro
- Sesiones protegidas
- Control de acceso por roles

### 🏷️ **Gestión de Categorías**
- ➕ Agregar nuevas categorías
- 🗑️ Eliminar categorías (con validación)
- 📊 Visualización de distribución por categoría
- ✅ Validación de productos existentes

### 📋 **Metadata de Productos**
- 📝 Campos personalizables de metadata
- 💾 Edición en tiempo real
- 🔍 Filtros por categoría y búsqueda
- 📸 Vista previa de imágenes
- ⚡ Guardado instantáneo

### 📊 **Dashboard Estadístico**
- 📈 Estadísticas completas del sistema
- 📊 Gráficos de distribución
- 🎯 Métricas de confianza
- 📱 Vista responsive

### ⚙️ **Configuración del Sistema**
- 🛠️ Parámetros del sistema
- 📝 Gestión de campos de metadata
- 💾 Backup automático
- 📤 Exportación de datos

### 💾 **Backup y Exportación**
- 📄 Exportar clasificaciones JSON
- 📋 Exportar metadata completa
- 💾 Backup completo del sistema
- 📅 Timestamp automático

## 🖥️ **Acceso al Panel**

### 🔗 URL de Acceso
```
http://localhost:5001/admin
```

### 🔑 Credenciales Administrativas
- **Usuario:** `admin`
- **Contraseña:** `clipadmin2025`

## 🎛️ **Funcionalidades por Sección**

### 1. 🏠 **Dashboard Principal**
- Vista general del sistema
- Estadísticas en tiempo real
- Navegación rápida
- Métricas de rendimiento

### 2. 🏷️ **Gestión de Categorías**
- **Agregar categorías:** Formulario dinámico
- **Eliminar categorías:** Con validación de productos
- **Vista de distribución:** Gráfico de barras
- **Conteo automático:** Productos por categoría

### 3. 📋 **Metadata de Productos**
- **Vista de catálogo:** Grid responsive
- **Edición in-line:** Formularios por producto
- **Filtros avanzados:** Por categoría y texto
- **Campos dinámicos:** Configurables desde settings

### 4. ⚙️ **Configuración**
- **Parámetros del sistema:** Límites y umbrales
- **Campos de metadata:** Agregar/eliminar campos
- **Tipos de datos:** Texto y números
- **Validación:** Campos requeridos/opcionales

## 📊 **Campos de Metadata Disponibles**

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| 🏷️ **codigo_producto** | Texto | ✅ Sí | Código único del producto |
| 💰 **precio** | Número | ❌ No | Precio del producto |
| 📝 **descripcion** | Texto | ❌ No | Descripción detallada |
| 📦 **stock** | Número | ❌ No | Cantidad en inventario |
| 👕 **talla_disponible** | Texto | ❌ No | Tallas disponibles |
| 🎨 **color** | Texto | ❌ No | Color del producto |

## 🔄 **API Endpoints**

### 🏷️ **Categorías**
```
POST   /admin/api/categories          # Agregar categoría
DELETE /admin/api/categories/{name}   # Eliminar categoría
```

### 📋 **Metadata**
```
POST   /admin/api/metadata/{filename} # Actualizar metadata
GET    /admin/api/search             # Buscar productos
```

### ⚙️ **Configuración**
```
POST   /admin/api/settings           # Guardar configuraciones
```

### 💾 **Backup**
```
GET    /admin/backup                 # Crear backup completo
GET    /admin/export/classifications # Exportar clasificaciones
GET    /admin/export/metadata        # Exportar metadata completa
```

## 📁 **Estructura de Archivos**

```
admin_panel.py              # Aplicación principal
admin_config.json           # Configuración del sistema
templates/
├── admin_login.html        # Página de login
├── admin_dashboard.html    # Dashboard principal
├── admin_categories.html   # Gestión de categorías
├── admin_metadata.html     # Gestión de metadata
└── admin_settings.html     # Configuración del sistema
```

## 🚀 **Instrucciones de Uso**

### 1. **Iniciar el Panel**
```bash
cd C:\Personal\CLIP_Comparador
python admin_panel.py
```

### 2. **Acceder al Sistema**
- Abrir navegador en `http://localhost:5001/admin`
- Ingresar credenciales administrativas
- Navegar por las diferentes secciones

### 3. **Gestionar Categorías**
- Ir a "Gestión de Categorías"
- Agregar nuevas categorías según necesidad
- Eliminar categorías sin productos asociados

### 4. **Administrar Metadata**
- Ir a "Metadata de Productos"
- Editar información de cada producto
- Usar filtros para navegar eficientemente

### 5. **Configurar Sistema**
- Ir a "Configuración"
- Ajustar parámetros del sistema
- Personalizar campos de metadata

### 6. **Crear Backups**
- Usar botón "Backup del Sistema" en dashboard
- O acceder desde "Configuración"
- Descargar exportaciones JSON

## 🛡️ **Seguridad**

- 🔐 **Autenticación obligatoria** para todas las funciones
- 🛡️ **Validación de datos** en formularios
- ⚠️ **Confirmaciones** para acciones destructivas
- 📝 **Logs de actividad** automáticos

## 🔄 **Integración con Sistema Principal**

El panel de administración se integra completamente con:
- 📂 **Catálogo de imágenes** (`catalogo/`)
- 🏷️ **Clasificaciones** (`product_classifications.json`)
- 🧠 **Embeddings** (`embeddings.json`)
- ⚙️ **Configuración** (`admin_config.json`)

## 🎯 **Próximas Mejoras**

- [ ] 📊 Reportes avanzados en PDF
- [ ] 📱 Aplicación móvil
- [ ] 🔔 Notificaciones push
- [ ] 📈 Analytics detallados
- [ ] 🌍 Multi-idioma
- [ ] 👥 Gestión de usuarios múltiples

---

**🏢 Sistema CLIP Comparador GOODY v2.0**  
*Panel de Administración de Metadata*