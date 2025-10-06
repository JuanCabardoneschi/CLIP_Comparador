"""
Rutas de autenticación
"""

from config.settings import lazy_import_heavy_deps
from auth.authentication import User

def create_auth_routes(app):
    """Crear las rutas de autenticación"""
    if not lazy_import_heavy_deps():
        raise Exception("No se pudieron cargar las dependencias")
    
    from flask import render_template, request, redirect, url_for, flash
    from flask_login import login_user, logout_user, current_user
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            print(f"Login attempt: username={username}, password={password}")
            
            # Credenciales simples para demo
            valid_credentials = [
                ('admin', 'admin123'),
                ('demo', 'clipdemo2025')
            ]
            
            if (username, password) in valid_credentials:
                user = User(username)
                login_success = login_user(user, remember=True)
                print(f"Login result: {login_success}")
                print(f"Current user after login: {current_user}")
                print(f"Is authenticated: {current_user.is_authenticated}")
                
                if login_success:
                    return redirect(url_for('index'))
                else:
                    flash('Error en el proceso de login')
            else:
                flash('Credenciales incorrectas')
        
        return render_template('login.html')
    
    @app.route('/logout')
    def logout():
        logout_user()
        return redirect(url_for('login'))