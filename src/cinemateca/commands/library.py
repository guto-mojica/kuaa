"""cinemateca library — registered film library operations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from cinemateca.commands._shared import _STEP_FULL_NAMES, print_banner, resolve_steps

app = typer.Typer(
    name="library",
    help="Operations across the registered film library (data/library/films.json).",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command("list")
def library_list(
    config: Annotated[
        Path | None,
        typer.Option(help="Caminho do arquivo config YAML."),
    ] = None,
) -> None:
    """List every registered film with its current per-film state."""
    from cinemateca.config import load_config, setup_logging
    from cinemateca.library import scan_library

    cfg = load_config(str(config) if config else None)
    setup_logging(cfg)
    films = scan_library(Path(cfg.paths.library_dir))

    if not films:
        print(f"Nenhum filme registrado em {Path(cfg.paths.library_dir)/'films.json'}")
        return

    print(f"{'SLUG':<50}  {'SCENES':>7}  {'PROCESSED':>10}  TITLE")
    print("─" * 100)
    for f in films:
        proc = "✓" if f.is_processed else "—"
        print(
            f"{f.slug:<50}  {f.scene_count:>7}  {proc:>10}  "
            f"{f.title}{f' ({f.year})' if f.year else ''}",
        )
    print(f"\n  {len(films)} filme(s) registrado(s)")


@app.command("reembed")
def library_reembed(
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only", help="Slug a processar (repetível). Padrão: todos os filmes registrados."
        ),
    ] = None,
    steps: Annotated[
        str,
        typer.Option(
            help="Etapas a executar, separadas por vírgula. "
            "Valores: frames, scenes, visual, embeddings, llm.",
        ),
    ] = "embeddings",
    keep_existing: Annotated[
        bool,
        typer.Option(
            "--keep-existing",
            help="Não apaga .npy / index_mapping.json antes de re-rodar.",
        ),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option(help="Caminho do arquivo config YAML."),
    ] = None,
) -> None:
    """Rebuild artifacts across every registered film (or a subset via --only)."""
    from cinemateca.config import load_config, setup_logging
    from cinemateca.library import scan_library
    from cinemateca.pipeline import CatalogPipeline

    cfg = load_config(str(config) if config else None)
    setup_logging(cfg)

    enabled = resolve_steps(steps)
    for step in _STEP_FULL_NAMES:
        setattr(cfg.pipeline.steps, step, step in enabled)

    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    if not films:
        typer.echo(f"✗ Nenhum filme registrado em {library_dir/'films.json'}", err=True)
        raise typer.Exit(1)

    only_set = set(only or [])
    if only_set:
        unknown = only_set - {f.slug for f in films}
        if unknown:
            typer.echo(f"✗ Slugs não registrados: {', '.join(sorted(unknown))}", err=True)
            raise typer.Exit(1)
        films = [f for f in films if f.slug in only_set]

    print_banner()
    print(f"  Filmes a reprocessar : {len(films)}", flush=True)
    print(f"  Etapas               : {','.join(sorted(enabled))}", flush=True)
    print(f"  Config               : {config or 'default'}", flush=True)
    print(f"  Apaga .npy antes     : {not keep_existing}\n", flush=True)

    raw_dir = Path(cfg.paths.raw_dir)
    summary: list[tuple[str, str, float]] = []
    for film in films:
        candidates = [film.raw_path, raw_dir / film.raw_path.name]
        video = next((p for p in candidates if p.exists()), None)
        if video is None:
            tried = ", ".join(str(p) for p in candidates)
            print(f"⏭  {film.slug} — raw não encontrado ({tried})\n", flush=True)
            summary.append((film.slug, "skipped (no raw)", 0.0))
            continue

        if not keep_existing:
            stale: list[tuple[str, tuple[str, ...]]] = []
            if "embeddings" in enabled:
                stale.append(("embeddings", ("keyframe_embeddings.npy", "index_mapping.json")))
            for subdir, files in stale:
                for fname in files:
                    p = library_dir / film.slug / subdir / fname
                    p.unlink(missing_ok=True)

        print(f"━━━ {film.slug} ━━━", flush=True)
        pipeline = CatalogPipeline(cfg, slug=film.slug)
        result = pipeline.run(str(video))
        status = "OK" if result.success else "FAIL"
        elapsed = float(getattr(result, "total_duration_s", 0.0))
        summary.append((film.slug, status, elapsed))
        print(flush=True)

    print("━" * 60, flush=True)
    print(f"  {'STATUS':<18}  {'TIME':>8}  SLUG", flush=True)
    for slug_, status, elapsed in summary:
        print(f"  {status:<18}  {elapsed:>6.1f}s  {slug_}", flush=True)
    n_ok = sum(1 for _, s, _ in summary if s == "OK")
    print(f"\n  {n_ok}/{len(summary)} success", flush=True)
    if n_ok != len(summary):
        raise typer.Exit(1)


@app.command("delete")
def library_delete(
    slug: Annotated[str, typer.Argument(help="Slug do filme a remover (ex: jeca_tatu).")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Confirma sem prompt interativo (use em scripts)."),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option(help="Caminho do arquivo config YAML."),
    ] = None,
) -> None:
    """Remove a film from the registry (and delete its on-disk artifacts)."""
    from cinemateca.config import load_config, setup_logging
    from cinemateca.library import delete_film, load_registry

    cfg = load_config(str(config) if config else None)
    setup_logging(cfg)
    library_dir = Path(cfg.paths.library_dir)

    registry = load_registry(library_dir)
    if slug not in registry:
        typer.echo(
            f"✗ Slug não registrado: {slug!r}. "
            f"Disponíveis: {', '.join(sorted(registry)) or '(nenhum)'}",
            err=True,
        )
        raise typer.Exit(1)

    if not yes and not typer.confirm(
        f"Remover {slug!r} e tudo em {library_dir/slug}? (irreversível)"
    ):
        typer.echo("Cancelado.")
        raise typer.Exit(0)

    delete_film(library_dir, slug=slug)
    film_dir = library_dir / slug
    if film_dir.exists():
        import shutil

        shutil.rmtree(film_dir)
    typer.echo(f"✓ {slug} removido.")
