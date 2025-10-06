import os
import sys
from waitress import serve
from app import app

if __name__ == '__main__':
    print('ðŸŒ Servidor WSGI iniciado con Waitress')
    print('ðŸ“ Acceder a: http://localhost:5000')
    print('â¹ï¸  Presionar Ctrl+C para detener')
    print('')
    
    # Configurar para producciÃ³n
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    # Iniciar servidor
    serve(
        app,
        host='0.0.0.0',
        port=5000,
        threads=8,
        connection_limit=1000,
        cleanup_interval=30,
        channel_timeout=120
    )
