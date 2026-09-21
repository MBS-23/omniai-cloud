"""Gunicorn settings for the container image."""

import gunicorn.http.wsgi

# Name the service, not the software serving it.
gunicorn.http.wsgi.SERVER = "OmniAI"

bind = "0.0.0.0:7860"

# One worker: every model is loaded once per process, so a second worker would
# double the memory for no gain on a CPU this size. Threads let a health check
# or a page load through while an analysis is running.
workers = 1
threads = 4
worker_class = "gthread"

# A long document can keep one request busy for minutes.
timeout = 600
graceful_timeout = 30

# Refuse oversized request lines and headers before the application sees them.
limit_request_line = 4096
limit_request_fields = 60
limit_request_field_size = 8190

accesslog = "-"
errorlog = "-"
