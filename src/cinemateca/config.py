"""
cinemateca.config
~~~~~~~~~~~~~~~~~
Carrega e valida a configuração do projeto a partir de arquivos YAML.

Uso típico:
    from cinemateca.config import load_config
    cfg = load_config("config/local.yaml")
    print(cfg.paths.metadata_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ─── Caminho do default embutido ──────────────────────────────────────────────
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "default.yaml"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Mescla recursivamente override em base. override tem precedência."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _resolve_paths(paths_dict: dict, project_root: Path) -> dict:
    """Converte strings de caminho relativo em Path absolutos."""
    resolved = {}
    for key, val in paths_dict.items():
        if isinstance(val, str):
            p = Path(val)
            resolved[key] = p if p.is_absolute() else project_root / p
        else:
            resolved[key] = val
    return resolved


# ─── Namespace simples para acesso com ponto ──────────────────────────────────

class _Namespace:
    """Permite cfg.section.key em vez de cfg['section']['key']."""

    def __init__(self, data: dict):
        for key, val in data.items():
            if isinstance(val, dict):
                setattr(self, key, _Namespace(val))
            else:
                setattr(self, key, val)

    def __repr__(self) -> str:
        return f"Namespace({vars(self)})"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, val in vars(self).items():
            result[key] = val.to_dict() if isinstance(val, _Namespace) else val
        return result


# ─── API pública ──────────────────────────────────────────────────────────────

def load_config(
    user_config: str | Path | None = None,
    project_root: str | Path | None = None,
) -> _Namespace:
    """
    Carrega a configuração, mesclando defaults com o arquivo do usuário.

    Args:
        user_config:  Caminho para config/local.yaml (opcional).
                      Se None, usa apenas os defaults.
        project_root: Raiz do projeto para resolver caminhos relativos.
                      Se None, usa o diretório de trabalho atual.

    Returns:
        _Namespace com toda a configuração acessível por atributos.

    Raises:
        FileNotFoundError: Se user_config for fornecido mas não existir.
        yaml.YAMLError:    Se algum YAML estiver malformado.
    """
    root = Path(project_root) if project_root else Path.cwd()

    # 1. Carregar defaults
    with open(_DEFAULT_CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Mesclar com config do usuário (se fornecida)
    if user_config is not None:
        user_path = Path(user_config)
        if not user_path.exists():
            raise FileNotFoundError(f"Config não encontrada: {user_path}")
        with open(user_path, encoding="utf-8") as f:
            user_data = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_data)
        logger.info("Config carregada: %s (sobre defaults)", user_path)
    else:
        logger.info("Usando config padrão (sem override do usuário)")

    # 3. Resolver caminhos relativos → absolutos
    config["paths"] = _resolve_paths(config.get("paths", {}), root)

    # 4. Criar diretórios necessários
    for path_obj in config["paths"].values():
        if isinstance(path_obj, Path):
            path_obj.mkdir(parents=True, exist_ok=True)

    return _Namespace(config)


def setup_logging(cfg: _Namespace) -> None:
    """
    Configura o sistema de logging com base na configuração.

    Deve ser chamado uma vez no início do pipeline ou da aplicação.
    """
    log_cfg = cfg.logging
    level = getattr(logging, log_cfg.level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_cfg.to_file:
        log_file = cfg.paths.logs_dir / log_cfg.filename
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logger.info("Logging inicializado — nível: %s", log_cfg.level)
