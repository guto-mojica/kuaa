"""kuaa serve — FastAPI web app launcher."""

from __future__ import annotations

from typing import Annotated

import typer

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.callback(invoke_without_command=True)
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
    import uvicorn

    uvicorn.run("api.server:app", host=host, port=port, reload=reload)
