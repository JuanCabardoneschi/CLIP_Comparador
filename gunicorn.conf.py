# Gunicorn configuration file for CLIP Comparador
import multiprocessing
import os

# Basic settings
bind = "0.0.0.0:5000"
workers = min(4, (multiprocessing.cpu_count() * 2) + 1)
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# Performance
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# Logging
errorlog = "-"
loglevel = "info"
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "clip_comparador"

# Development vs Production
if os.getenv('FLASK_ENV') == 'development':
    reload = True
    loglevel = "debug"
else:
    reload = False
    
# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Worker settings for ML models
worker_tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
tmp_upload_dir = None