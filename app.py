"""Legacy FastAPI entrypoint — kept for back-compat with ``uv run app.py``.

The canonical invocation is now::

    kuaa serve [--host ...] [--port ...] [--reload/--no-reload]

which is auto-discoverable via ``kuaa --help``. This file just
delegates so existing muscle memory and docs that still say
``uv run app.py`` keep working until they're updated.
"""

from kuaa.__main__ import serve

if __name__ == "__main__":
    # Match the historical defaults of this script (host 127.0.0.1, port
    # 8501, reload=True). Users who want non-defaults should use
    # ``kuaa serve --port ...`` directly.
    serve(host="127.0.0.1", port=8501, reload=True)
