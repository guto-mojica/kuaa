"""
cinemateca.__main__
~~~~~~~~~~~~~~~~~~~
Unified CLI entry point (Typer).

Command tree:
    cinemateca serve                  # FastAPI web app
    cinemateca info VIDEO             # Video properties
    cinemateca process VIDEO ...      # Full pipeline (single film)
    cinemateca library list           # Show registered films + state
    cinemateca library reembed ...    # Rebuild embeddings across registry
    cinemateca library delete SLUG    # Remove a film + artifacts
    cinemateca config show            # Dump effective merged config
    cinemateca eval seed ...          # Seed sample queries for /eval

Each command family lives in ``src/cinemateca/commands/``.
"""

from __future__ import annotations

import typer

from cinemateca.commands import config_cmd, eval_cmd, library, process

app = typer.Typer(
    name="cinemateca",
    help="Cinemateca AI — offline audiovisual cataloguing for film archives.",
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
    import os
    import sys
    import uvicorn
    from pathlib import Path

    # api/ is not an installed package (only src/ is in pyproject packages.find).
    # Add the project root so uvicorn can import "api.server" both in the main
    # process and in reload subprocesses (which inherit PYTHONPATH, not sys.path).
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    existing_pp = os.environ.get("PYTHONPATH", "")
    if project_root not in existing_pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            f"{project_root}{os.pathsep}{existing_pp}" if existing_pp else project_root
        )

    uvicorn.run("api.server:app", host=host, port=port, reload=reload)


# Register command families as sub-apps.
app.add_typer(library.app, name="library")
app.add_typer(config_cmd.app, name="config")
app.add_typer(eval_cmd.app, name="eval")

# Backward-compat aliases for the old _resolve_steps / _print_banner used in
# any external script that imported from __main__ directly.
from cinemateca.commands._shared import print_banner as _print_banner  # noqa: E402
from cinemateca.commands._shared import resolve_steps as _resolve_steps  # noqa: E402
from cinemateca.commands._shared import _STEP_ALIASES, _STEP_FULL_NAMES  # noqa: E402


def main() -> None:
    """Console-script entry (``[project.scripts]`` → ``cinemateca``)."""
    app()


if __name__ == "__main__":
    main()
