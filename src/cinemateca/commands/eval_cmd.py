"""cinemateca eval — eval set builder utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="eval",
    help="Eval set builder utilities (seed sample queries, etc.).",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command("seed")
def eval_seed(
    run: Annotated[
        str,
        typer.Option(help="Run ID (becomes <run>.queries.json in the eval root)."),
    ] = "default",
    queries: Annotated[
        int,
        typer.Option(help="Number of sample queries to write (clamped to the bundled max).", min=0),
    ] = 5,
    root: Annotated[
        Path,
        typer.Option(help="Eval data directory. Created if it doesn't exist."),
    ] = Path("data/eval"),
) -> None:
    """Write a sample queries file for the /eval grading UI."""
    from cinemateca.eval.seed import SAMPLE_QUERIES, write_seed

    path = write_seed(root=root, run_id=run, count=queries)
    written = min(max(0, queries), len(SAMPLE_QUERIES))
    typer.echo(f"✓ Wrote {written} queries to {path}")
