# 🔍 CLIP Comparador GOODY - Sistema Completo de Búsqueda Visual# 🔍 CLIP Comparador GOODY - Sistema Completo de Búsqueda Visual



## 📋 Descripción del Sistema## 📋 Descripción del Sistema



Sistema completo de búsqueda visual inteligente con **dos interfaces principales**:Sistema completo de búsqueda visual inteligente con **dos interfaces principales**:

- 🛍️ **Interface Cliente/Demo** - Para usuarios finales y demostraciones- 🛍️ **Interface Cliente/Demo** - Para usuarios finales y demostraciones

- 🔧 **Panel Administrativo** - Para gestión de metadata y configuraciones- 🔧 **Panel Administrativo** - Para gestión de metadata y configuraciones



## 🚀 Características Principales## 🚀 Características Principales



### 🛍️ **Sistema Cliente (Puerto 5000)**### 🛍️ **Sistema Cliente (Puerto 5000)**

- 🔍 Búsqueda visual con tecnología CLIP- 🔍 Búsqueda visual con tecnología CLIP

- 📱 Interface responsive y moderna- 📱 Interface responsive y moderna

- ⚡ Resultados instantáneos con similitud- ⚡ Resultados instantáneos con similitud

- 🎯 Top-5 productos más similares- 🎯 Top-5 productos más similares

- 📊 Métricas de confianza visual- 📊 Métricas de confianza visual



### 🔧 **Panel Administrativo (Puerto 5001)**### 🔧 **Panel Administrativo (Puerto 5001)**

- 🔐 Sistema de autenticación seguro- 🔐 Sistema de autenticación seguro

- 🏷️ Gestión completa de categorías- 🏷️ Gestión completa de categorías

- 📋 Administración de metadata de productos- 📋 Administración de metadata de productos

- 📊 Dashboard con estadísticas en tiempo real- 📊 Dashboard con estadísticas en tiempo real

- 💾 Sistema de backup y exportación- 💾 Sistema de backup y exportación

- ⚙️ Configuración avanzada del sistema- ⚙️ Configuración avanzada del sistema



## 🎯 **Accesos al Sistema**## 🎯 **Accesos al Sistema**



### 🛍️ **Interface Cliente/Demo**### 🛍️ **Interface Cliente/Demo**

``````

URL: http://localhost:5000URL: http://localhost:5000

Acceso: Público (sin autenticación)Acceso: Público (sin autenticación)

Propósito: Búsqueda visual para usuarios finalesPropósito: Búsqueda visual para usuarios finales

``````



### 🔧 **Panel Administrativo**### 🔧 **Panel Administrativo**

``````

URL: http://localhost:5001/adminURL: http://localhost:5001/admin

Usuario: adminUsuario: admin

Contraseña: clipadmin2025Contraseña: clipadmin2025

Propósito: Gestión de metadata y configuracionesPropósito: Gestión de metadata y configuraciones

``````



## 🏗️ **Arquitectura del Sistema**## 🏗️ **Arquitectura del Sistema**



``````

CLIP_Comparador/CLIP_Comparador/

├── 🛍️ SISTEMA CLIENTE├── 🛍️ SISTEMA CLIENTE

│   ├── app_simple.py              # Aplicación principal (Puerto 5000)│   ├── app_simple.py              # Aplicación principal (Puerto 5000)

│   ├── generate_embeddings.py     # Generador de embeddings CLIP│   ├── generate_embeddings.py     # Generador de embeddings CLIP

│   ├── auto_preclassify.py        # Pre-clasificación automática│   ├── auto_preclassify.py        # Pre-clasificación automática

│   └── manual_classifier.py       # Clasificador manual│   └── manual_classifier.py       # Clasificador manual

││

├── 🔧 PANEL ADMINISTRATIVO├── 🔧 PANEL ADMINISTRATIVO

│   ├── admin_panel.py             # Panel admin (Puerto 5001)│   ├── admin_panel.py             # Panel admin (Puerto 5001)

│   ├── admin_config.json          # Configuración del sistema│   ├── admin_config.json          # Configuración del sistema

│   └── templates/admin_*.html     # Templates administrativos│   └── templates/admin_*.html     # Templates administrativos

││

├── 📂 DATOS Y CONFIGURACIÓN├── 📂 DATOS Y CONFIGURACIÓN

│   ├── catalogo/                  # Imágenes del catálogo│   ├── catalogo/                  # Imágenes del catálogo

│   ├── embeddings.json           # Cache de embeddings│   ├── embeddings.json           # Cache de embeddings

│   ├── product_classifications.json # Clasificaciones│   ├── product_classifications.json # Clasificaciones

│   └── requirements.txt           # Dependencias│   └── requirements.txt           # Dependencias

││

└── 📋 DOCUMENTACIÓN└── 📋 DOCUMENTACIÓN

    ├── README.md                  # Documentación principal    ├── README.md                  # Documentación principal

    └── ADMIN_README.md           # Documentación del panel admin    └── ADMIN_README.md           # Documentación del panel admin

``````



## 🚀 **Instalación y Configuración**## 🚀 **Instalación y Configuración**



### 1. **Clonar Repositorio**### 1. **Clonar Repositorio**

```bash```bash

git clone https://github.com/JuanCabardoneschi/CLIP_Comparador.gitgit clone https://github.com/JuanCabardoneschi/CLIP_Comparador.git

cd CLIP_Comparadorcd CLIP_Comparador

``````



### 2. **Configurar Entorno Virtual**### 2. **Configurar Entorno Virtual**

```bash```bash

python -m venv venvpython -m venv venv

venv\Scripts\activate  # Windowsvenv\Scripts\activate  # Windows

pip install -r requirements.txtpip install -r requirements.txt

``````



### 3. **Generar Embeddings del Catálogo**### 3. **Generar Embeddings del Catálogo**

```bash```bash

python generate_embeddings.pypython generate_embeddings.py

``````



### 4. **Pre-clasificar Productos (Opcional)**### 4. **Pre-clasificar Productos (Opcional)**

```bash```bash

python auto_preclassify.pypython auto_preclassify.py

``````



## 🖥️ **Ejecutar el Sistema**## 🖥️ **Ejecutar el Sistema**



### 🛍️ **Iniciar Sistema Cliente**### 🛍️ **Iniciar Sistema Cliente**

```bash```bash

python app_simple.pypython app_simple.py

# Acceder a: http://localhost:5000# Acceder a: http://localhost:5000

``````



### 🔧 **Iniciar Panel Administrativo**### 🔧 **Iniciar Panel Administrativo**

```bash```bash

python admin_panel.pypython admin_panel.py

# Acceder a: http://localhost:5001/admin# Acceder a: http://localhost:5001/admin

``````



### 🔄 **Ejecutar Ambos Sistemas (Recomendado)**### 🔄 **Ejecutar Ambos Sistemas (Recomendado)**

```bash```bash

# Terminal 1# Terminal 1

python app_simple.pypython app_simple.py



# Terminal 2# Terminal 2

python admin_panel.pypython admin_panel.py

``````



## 🏷️ **Categorías de Productos Soportadas**## 🏷️ **Categorías de Productos Soportadas**



| Categoría | Productos | Interface || Categoría | Productos | Interface |

|-----------|-----------|-----------||-----------|-----------|-----------|

| 🥼 **DELANTAL** | Delantales, mandiles | Cliente + Admin || 🥼 **DELANTAL** | Delantales, mandiles | Cliente + Admin |

| 👔 **CAMISAS** | Camisas profesionales | Cliente + Admin || 👔 **CAMISAS** | Camisas profesionales | Cliente + Admin |

| 🧥 **CHAQUETAS** | Chaquetas chef, profesionales | Cliente + Admin || 🧥 **CHAQUETAS** | Chaquetas chef, profesionales | Cliente + Admin |

| 👔 **AMBO VESTIR** | Uniformes médicos | Cliente + Admin || 👔 **AMBO VESTIR** | Uniformes médicos | Cliente + Admin |

| 🧥 **CASACAS** | Casacas profesionales | Cliente + Admin || 🧥 **CASACAS** | Casacas profesionales | Cliente + Admin |

| 👟 **ZUECOS** | Calzado profesional | Cliente + Admin || 👟 **ZUECOS** | Calzado profesional | Cliente + Admin |

| 🧢 **GORROS** | Gorras, sombreros | Cliente + Admin || 🧢 **GORROS** | Gorras, sombreros | Cliente + Admin |

| 🧥 **CARDIGAN** | Cardigans profesionales | Cliente + Admin || 🧥 **CARDIGAN** | Cardigans profesionales | Cliente + Admin |

| 👔 **BUZOS** | Buzos de trabajo | Cliente + Admin || 👔 **BUZOS** | Buzos de trabajo | Cliente + Admin |

| 👠 **ZAPATO DAMA** | Calzado femenino | Cliente + Admin || 👠 **ZAPATO DAMA** | Calzado femenino | Cliente + Admin |

| 🦺 **CHALECO** | Chalecos profesionales | Cliente + Admin || 🦺 **CHALECO** | Chalecos profesionales | Cliente + Admin |

| 👕 **REMERAS** | Camisetas, polos | Cliente + Admin || 👕 **REMERAS** | Camisetas, polos | Cliente + Admin |



## 📊 **Funcionalidades por Interface**## 📊 **Funcionalidades por Interface**



### 🛍️ **Sistema Cliente**### 🛍️ **Sistema Cliente**

- ✅ Búsqueda por imagen upload- ✅ Búsqueda por imagen upload

- ✅ Resultados con similitud visual- ✅ Resultados con similitud visual

- ✅ Filtros por categoría- ✅ Filtros por categoría

- ✅ Interface responsive- ✅ Interface responsive

- ✅ Métricas de rendimiento- ✅ Métricas de rendimiento

- ✅ Estado del sistema en tiempo real- ✅ Estado del sistema en tiempo real



### 🔧 **Panel Administrativo**### 🔧 **Panel Administrativo**

- ✅ Dashboard con estadísticas- ✅ Dashboard con estadísticas

- ✅ Gestión de categorías- ✅ Gestión de categorías

- ✅ Administración de metadata- ✅ Administración de metadata

- ✅ Configuración del sistema- ✅ Configuración del sistema

- ✅ Backup y exportación- ✅ Backup y exportación

- ✅ Sistema de autenticación- ✅ Sistema de autenticación

- ✅ Reportes y analytics- ✅ Reportes y analytics



## 📋 **Campos de Metadata Administrables**## 📋 **Campos de Metadata Administrables**



| Campo | Tipo | Requerido | Descripción || Campo | Tipo | Requerido | Descripción |

|-------|------|-----------|-------------||-------|------|-----------|-------------|

| 🏷️ **codigo_producto** | Texto | ✅ | Código único del producto || 🏷️ **codigo_producto** | Texto | ✅ | Código único del producto |

| 💰 **precio** | Número | ❌ | Precio del producto || 💰 **precio** | Número | ❌ | Precio del producto |

| 📝 **descripcion** | Texto | ❌ | Descripción detallada || 📝 **descripcion** | Texto | ❌ | Descripción detallada |

| 📦 **stock** | Número | ❌ | Cantidad en inventario || 📦 **stock** | Número | ❌ | Cantidad en inventario |

| 👕 **talla_disponible** | Texto | ❌ | Tallas disponibles || 👕 **talla_disponible** | Texto | ❌ | Tallas disponibles |

| 🎨 **color** | Texto | ❌ | Color del producto || 🎨 **color** | Texto | ❌ | Color del producto |



## 🔄 **API Endpoints**## 🔄 **API Endpoints**



### 🛍️ **API Cliente**### 🛍️ **API Cliente**

``````

POST /search          # Búsqueda por imagenPOST /search          # Búsqueda por imagen

POST /search_text     # Búsqueda por textoPOST /search_text     # Búsqueda por texto

GET  /health          # Estado del sistemaGET  /health          # Estado del sistema

``````



### 🔧 **API Administrativa**### 🔧 **API Administrativa**

``````

POST /admin/api/categories          # Gestión de categoríasPOST /admin/api/categories          # Gestión de categorías

POST /admin/api/metadata/{filename} # Actualizar metadataPOST /admin/api/metadata/{filename} # Actualizar metadata

GET  /admin/api/search             # Buscar productosGET  /admin/api/search             # Buscar productos

POST /admin/api/settings           # ConfiguracionesPOST /admin/api/settings           # Configuraciones

GET  /admin/backup                 # Crear backupGET  /admin/backup                 # Crear backup

``````



## 🛡️ **Seguridad**## 🛡️ **Seguridad**



- 🔐 **Panel Admin**: Autenticación obligatoria- 🔐 **Panel Admin**: Autenticación obligatoria

- 🛡️ **Validación**: Datos de entrada validados- 🛡️ **Validación**: Datos de entrada validados

- 📝 **Logs**: Actividad del sistema registrada- 📝 **Logs**: Actividad del sistema registrada

- 💾 **Backup**: Sistema automático de respaldo- 💾 **Backup**: Sistema automático de respaldo



## 🚀 **Casos de Uso**## 🚀 **Casos de Uso**



### 👥 **Para Usuarios Finales**### 👥 **Para Usuarios Finales**

1. Subir imagen de producto buscado1. Subir imagen de producto buscado

2. Obtener productos similares instantáneamente2. Obtener productos similares instantáneamente

3. Explorar catálogo por categorías3. Explorar catálogo por categorías

4. Ver métricas de similitud4. Ver métricas de similitud



### 👨‍💼 **Para Administradores**### 👨‍💼 **Para Administradores**

1. Gestionar categorías de productos1. Gestionar categorías de productos

2. Administrar metadata detallada2. Administrar metadata detallada

3. Configurar parámetros del sistema3. Configurar parámetros del sistema

4. Generar reportes y backups4. Generar reportes y backups

5. Monitorear estadísticas en tiempo real5. Monitorear estadísticas en tiempo real



## 📈 **Métricas y Rendimiento**## 📈 **Métricas y Rendimiento**



- ⚡ **Búsqueda**: < 2 segundos promedio- ⚡ **Búsqueda**: < 2 segundos promedio

- 🎯 **Precisión**: 85%+ en categorización- 🎯 **Precisión**: 85%+ en categorización

- 📊 **Cobertura**: 12 categorías principales- 📊 **Cobertura**: 12 categorías principales

- 💾 **Cache**: Embeddings pre-calculados- 💾 **Cache**: Embeddings pre-calculados

- 🔄 **Actualización**: Tiempo real- 🔄 **Actualización**: Tiempo real



## 🔧 **Tecnologías Utilizadas**## 🔧 **Tecnologías Utilizadas**



- 🧠 **CLIP**: OpenAI CLIP para embeddings visuales- 🧠 **CLIP**: OpenAI CLIP para embeddings visuales

- 🐍 **Python**: Backend y procesamiento- 🐍 **Python**: Backend y procesamiento

- 🌐 **Flask**: Framework web- 🌐 **Flask**: Framework web

- 🎨 **HTML/CSS/JS**: Frontend responsive- 🎨 **HTML/CSS/JS**: Frontend responsive

- 📊 **NumPy**: Cálculos de similitud- 📊 **NumPy**: Cálculos de similitud

- 🖼️ **PIL**: Procesamiento de imágenes- 🖼️ **PIL**: Procesamiento de imágenes

- 📱 **Bootstrap**: Interface moderna- 📱 **Bootstrap**: Interface moderna



## 📝 **Próximas Mejoras**## 📝 **Próximas Mejoras**



- [ ] 🔍 Búsqueda por múltiples imágenes- [ ] 🔍 Búsqueda por múltiples imágenes

- [ ] 📱 App móvil nativa- [ ] 📱 App móvil nativa

- [ ] 🌍 Soporte multi-idioma- [ ] 🌍 Soporte multi-idioma

- [ ] 📊 Analytics avanzados- [ ] 📊 Analytics avanzados

- [ ] 🤖 IA para descripción automática- [ ] 🤖 IA para descripción automática

- [ ] 🔔 Notificaciones push- [ ] 🔔 Notificaciones push

- [ ] 👥 Sistema multi-usuario- [ ] 👥 Sistema multi-usuario



## 🤝 **Contribución**## 🤝 **Contribución**



1. Fork el proyecto1. Fork el proyecto

2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)

3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)

4. Push al branch (`git push origin feature/AmazingFeature`)4. Push al branch (`git push origin feature/AmazingFeature`)

5. Abre un Pull Request5. Abre un Pull Request



## 📄 **Licencia**## 📄 **Licencia**



Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.



## 👤 **Autor**## 👤 **Autor**



**Juan Cabardoneschi****Juan Cabardoneschi**

- GitHub: [@JuanCabardoneschi](https://github.com/JuanCabardoneschi)- GitHub: [@JuanCabardoneschi](https://github.com/JuanCabardoneschi)



## 📞 **Soporte**## 📞 **Soporte**



Para soporte técnico o consultas:Para soporte técnico o consultas:

- 📧 Email: [contacto](mailto:tu-email@ejemplo.com)- 📧 Email: [contacto](mailto:tu-email@ejemplo.com)

- 💬 Issues: [GitHub Issues](https://github.com/JuanCabardoneschi/CLIP_Comparador/issues)- 💬 Issues: [GitHub Issues](https://github.com/JuanCabardoneschi/CLIP_Comparador/issues)



------



**🏢 Sistema CLIP Comparador GOODY v2.0**  **🏢 Sistema CLIP Comparador GOODY v2.0**  

*Búsqueda Visual Inteligente + Panel Administrativo Completo**Búsqueda Visual Inteligente + Panel Administrativo Completo*