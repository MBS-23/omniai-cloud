"""Entry point.

    python -m backend.main

Kept as a thin shim so the command documented in the thesis still works; the
application itself is built by :func:`backend.app.create_app`.
"""

from __future__ import annotations

from backend.app import create_app
from backend.config import get_settings

settings = get_settings()

# Module-level app object, for `gunicorn backend.main:app`.
app = create_app(settings)


def main() -> None:
    app.run(host=settings.host, port=settings.port, debug=settings.debug, threaded=True)


if __name__ == "__main__":
    main()
