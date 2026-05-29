"""Config loading + logging setup (F1).

Preserves the historical ``load_config`` signature and merge semantics
(default.yaml ⊕ user override, relative→absolute path resolution,
opt-out dir creation). The merged mapping is parsed into the typed
:class:`Settings` model; a schema violation becomes a
:class:`cinemateca.errors.ConfigError` whose message names the offending
field path instead of surfacing a deep ``AttributeError`` at use time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from cinemateca.config.schema import CONFIG_VERSION, Settings
from cinemateca.errors import ConfigError

logger = logging.getLogger(__name__)

# ─── Caminho do default embutido ──────────────────────────────────────────────
# NB: one extra ``.parent`` vs. the old module — config.py was at
# src/cinemateca/config.py; loader.py is at src/cinemateca/config/loader.py.
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"


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


def _ensure_dirs(cfg: Settings) -> None:
    """Create every configured ``paths.*`` directory on disk."""
    for path_obj in cfg.paths.model_dump(mode="python").values():
        if isinstance(path_obj, Path):
            path_obj.mkdir(parents=True, exist_ok=True)


# ─── API pública ──────────────────────────────────────────────────────────────


def load_config(
    user_config: str | Path | None = None,
    project_root: str | Path | None = None,
    *,
    ensure_dirs: bool = True,
) -> Settings:
    """
    Carrega a configuração, mesclando defaults com o arquivo do usuário.

    Args:
        user_config:  Caminho para config/local.yaml (opcional).
                      Se None, usa apenas os defaults.
        project_root: Raiz do projeto para resolver caminhos relativos.
                      Se None, usa o diretório de trabalho atual.
        ensure_dirs:  When ``True`` (default), create every ``paths.*``
                      directory that does not yet exist.  Pass ``False``
                      in tests or introspection-only callers to avoid
                      filesystem side-effects on load.

    Returns:
        :class:`Settings` com toda a configuração acessível por atributos.

    Raises:
        FileNotFoundError: Se user_config for fornecido mas não existir.
        ConfigError:       Se o YAML mesclado violar o schema tipado.
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

    # 4. Versão do schema (default.yaml já a traz; manter back-compat).
    config.setdefault("config_version", CONFIG_VERSION)

    # 5. Validar contra o schema tipado; erro de schema → ConfigError com path.
    try:
        settings = Settings.model_validate(config)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        raise ConfigError(
            f"Invalid configuration at '{loc}': {first['msg']}",
            code="config.invalid",
        ) from exc

    # 6. Criar diretórios necessários (opt-out for tests / introspection)
    if ensure_dirs:
        _ensure_dirs(settings)

    return settings


def setup_logging(cfg: Settings) -> None:
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


__all__: list[str] = ["load_config", "setup_logging"]
