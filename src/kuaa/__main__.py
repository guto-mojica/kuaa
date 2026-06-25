"""
kuaa.__main__
~~~~~~~~~~~~~~~~~~~
Unified CLI entry point (Typer).

Command tree:
    kuaa serve                  # FastAPI web app
    kuaa info VIDEO             # Video properties
    kuaa process VIDEO ...      # Full pipeline (single film)
    kuaa library list           # Show registered films + state
    kuaa library reembed ...    # Rebuild embeddings across registry
    kuaa library delete SLUG    # Remove a film + artifacts
    kuaa config show            # Dump effective merged config
    kuaa eval seed ...          # Seed sample queries for /eval

Each command family lives in ``src/kuaa/commands/``.
"""

from __future__ import annotations

import typer

from kuaa.commands import config_cmd, eval_cmd, library, process

app = typer.Typer(
    name="kuaa",
    help="KUAA — knowledge from unstructured audiovisual archives.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Register top-level commands from process sub-module.
app.command("info")(process.info)
app.command("process")(process.process)

# Register serve inline (thin — no sub-app needed).
from typing import Annotated  # noqa: E402 — kept close to use site


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8501,
    reload: Annotated[
        bool,
        typer.Option("--reload/--no-reload", help="Auto-reload on code changes (dev mode)."),
    ] = True,
) -> None:
    """Run the FastAPI web app (the v0.3+ UI).

    Replaces the legacy ``uv run app.py`` invocation. Opens
    ``http://<host>:<port>``.
    """
    from pathlib import Path

    import uvicorn

    project_root = str(Path(__file__).parent.parent.parent)
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        app_dir=project_root,
    )


# Register command families as sub-apps.
app.add_typer(library.app, name="library")
app.add_typer(config_cmd.app, name="config")
app.add_typer(eval_cmd.app, name="eval")

# Backward-compat aliases for the old _resolve_steps / _print_banner used in
# any external script that imported from __main__ directly.
from kuaa.commands._shared import (  # noqa: E402,F401,I001
    _STEP_ALIASES,
    _STEP_FULL_NAMES,
    print_banner as _print_banner,
    resolve_steps as _resolve_steps,
)


def main() -> None:
    """Console-script entry (``[project.scripts]`` → ``kuaa``)."""
    app()


if __name__ == "__main__":
    main()
