"""cinemateca config — configuration inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="config",
    help="Inspect the merged effective configuration (default.yaml ⊕ local.yaml).",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command("show")
def config_show(
    config: Annotated[
        Path | None,
        typer.Option(help="Caminho do arquivo config YAML."),
    ] = None,
) -> None:
    """Dump the effective merged config (default.yaml ⊕ local.yaml ⊕ --config)."""
    import yaml

    from cinemateca.config import load_config

    cfg = load_config(str(config) if config else None)

    def _serialise(obj):
        if hasattr(obj, "__dict__"):
            return {k: _serialise(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(v) for v in obj]
        return obj

    print(yaml.safe_dump(_serialise(cfg), sort_keys=True, allow_unicode=True))
