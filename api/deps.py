"""FastAPI dependency providers."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_config():
    from cinemateca.config import load_config

    local = Path("config/local.yaml")
    return load_config(str(local) if local.exists() else None)
