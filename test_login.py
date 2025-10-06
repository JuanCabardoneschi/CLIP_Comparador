"""
Script para probar el login del sistema
"""

try:
    import requests
    from requests.sessions import Session
    print("Imports exitosos")
except Exception as e:
    print(f"Error importando: {e}")
    exit(1)

def test_login():
    try:
        # Crear una sesión para mantener cookies
        session = Session()
        base_url = "http://127.0.0.1:5000"
        
        # 1. Acceder a la página de login primero para obtener cookies
        print("1. Accediendo a la página de login...")
        response = session.get(f"{base_url}/login")
        print(f"Status: {response.status_code}")
        print(f"Cookies: {response.cookies}")
        
        # 2. Hacer POST con credenciales
        print("\n2. Enviando credenciales...")
        login_data = {
            'username': 'admin',
            'password': 'admin123'
        }
        response = session.post(f"{base_url}/login", data=login_data, allow_redirects=False)
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Cookies después del login: {session.cookies}")
        
        # 3. Verificar si hay redirección
        if response.status_code == 302:
            print(f"Redirección a: {response.headers.get('Location')}")
            
            # 4. Seguir la redirección manualmente
            print("\n3. Siguiendo redirección...")
            response = session.get(f"{base_url}/")
            print(f"Status final: {response.status_code}")
            if response.status_code == 200:
                print("✓ Login exitoso - acceso a página principal")
            else:
                print("✗ Error accediendo a página principal")
        else:
            print("✗ No hubo redirección después del login")
            print(f"Response content: {response.text[:200]}...")
    except Exception as e:
        print(f"Error durante el test: {e}")

if __name__ == "__main__":
    print("Iniciando test de login...")
    test_login()