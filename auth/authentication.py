"""
Sistema de autenticación
"""

from config.settings import lazy_import_heavy_deps

class User:
    def __init__(self, user_id):
        self.id = user_id
    
    @property
    def is_authenticated(self):
        return True
    
    @property
    def is_active(self):
        return True
    
    @property
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)

def setup_login_manager(app):
    """Configurar el LoginManager"""
    if not lazy_import_heavy_deps():
        raise Exception("No se pudieron cargar las dependencias")
    
    from flask_login import LoginManager
    
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Por favor inicia sesión para acceder.'
    
    @login_manager.user_loader
    def load_user(user_id):
        # Usuarios válidos
        valid_users = ['admin', 'demo']
        return User(user_id) if user_id in valid_users else None
    
    return login_manager