"""kuaa library — registered film library operations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from kuaa.commands._shared import _STEP_FULL_NAMES, print_banner, resolve_steps

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
    from kuaa.config import load_config, setup_logging
    from kuaa.library import scan_library

    cfg = load_config(str(config) if config else None)
    setup_logging(cfg)
    films = scan_library(Path(cfg.paths.library_dir))

    if not films:
        print(f"Nenhum filme registrado em {Path(cfg.paths.library_dir) / 'films.json'}")
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
    from kuaa.config import load_config, setup_logging
    from kuaa.library import scan_library
    from kuaa.pipeline import CatalogPipeline

    cfg = load_config(str(config) if config else None)
    setup_logging(cfg)

    enabled = resolve_steps(steps)
    for step in _STEP_FULL_NAMES:
        setattr(cfg.pipeline.steps, step, step in enabled)

    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    if not films:
        typer.echo(f"✗ Nenhum filme registrado em {library_dir / 'films.json'}", err=True)
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


@app.command("reindex-vectors")
def library_reindex_vectors(
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only", help="Slug a reindexar (repetível). Padrão: todos os filmes registrados."
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(help="Caminho do arquivo config YAML."),
    ] = None,
) -> None:
    """Rebuild the configured vector index from each film's existing embeddings.

    Idempotent maintenance command: reads the already-computed
    ``keyframe_embeddings.npy`` + ``index_mapping.json`` pair already on disk
    for each film — no model inference — and upserts it into whichever
    backend ``search.index_backend`` currently selects. No-op for the default
    ``numpy_bruteforce`` backend, whose store already IS that ``.npy`` pair.

    Use this after switching to ``lancedb`` (already-processed films are not
    backfilled automatically), after a missing/corrupted index directory, or
    to repair drift from a best-effort upsert failure during normal
    processing (``_maybe_index_embeddings`` logs and continues on error
    rather than failing the pipeline run).
    """
    from kuaa.config import load_config, setup_logging
    from kuaa.library import FilmContext, scan_library
    from kuaa.retrieval.vector_index import get_vector_index
    from kuaa.search.cache import IndexStatus, load_index

    cfg = load_config(str(config) if config else None)
    setup_logging(cfg)

    backend = getattr(cfg.search, "index_backend", "numpy_bruteforce")
    print_banner()
    if backend == "numpy_bruteforce":
        print(
            "  index_backend = numpy_bruteforce — nada a fazer "
            "(o .npy de cada filme já É o índice)."
        )
        return

    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    if not films:
        typer.echo(f"✗ Nenhum filme registrado em {library_dir / 'films.json'}", err=True)
        raise typer.Exit(1)

    only_set = set(only or [])
    if only_set:
        unknown = only_set - {f.slug for f in films}
        if unknown:
            typer.echo(f"✗ Slugs não registrados: {', '.join(sorted(unknown))}", err=True)
            raise typer.Exit(1)
        films = [f for f in films if f.slug in only_set]

    print(f"  Backend              : {backend}")
    print(f"  Filmes candidatos    : {len(films)}\n", flush=True)

    summary: list[tuple[str, str, int]] = []
    for film in films:
        ctx = FilmContext.for_film(cfg, film.slug)
        index = load_index(
            ctx,
            mapping_filename=cfg.embeddings.mapping_filename,
            embeddings_filename=cfg.embeddings.filename,
            cfg=cfg,
        )
        if index.status is IndexStatus.MISSING:
            print(f"  ⏭  {film.slug} — sem embeddings ainda ({index.detail})", flush=True)
            summary.append((film.slug, "skipped (missing)", 0))
            continue
        if index.status is IndexStatus.CORRUPT:
            print(f"  ✗ {film.slug} — índice corrompido ({index.detail})", flush=True)
            summary.append((film.slug, "skipped (corrupt)", 0))
            continue

        # index.ok (equivalent to status is OK here) guarantees embeddings/kf_df
        # are non-None — narrows the Optional fields for mypy, mirroring
        # src/kuaa/search/clip.py's convention.
        assert index.embeddings is not None
        assert index.kf_df is not None
        cols = [c for c in ("scene_id", "keyframe_id", "filepath") if c in index.kf_df.columns]
        rows = index.kf_df[cols].copy().reset_index(drop=True)
        rows["film_slug"] = film.slug
        vector_index = get_vector_index(cfg)
        # Delete-then-add makes this idempotent: re-running (drift repair,
        # disaster recovery) replaces the film's rows instead of duplicating
        # them on every run.
        vector_index.delete(film.slug)
        vector_index.add(index.embeddings, rows)
        print(f"  ✓ {film.slug} — {len(rows)} vetores", flush=True)
        summary.append((film.slug, "indexed", len(rows)))

    print("━" * 60, flush=True)
    n_ok = sum(1 for _, s, _ in summary if s == "indexed")
    n_vectors = sum(n for _, s, n in summary if s == "indexed")
    print(
        f"  {n_ok}/{len(summary)} filmes indexados · {n_vectors} vetores · backend={backend}",
        flush=True,
    )


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
    from kuaa.config import load_config, setup_logging
    from kuaa.library import delete_film, load_registry

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
        f"Remover {slug!r} e tudo em {library_dir / slug}? (irreversível)"
    ):
        typer.echo("Cancelado.")
        raise typer.Exit(0)

    delete_film(library_dir, slug=slug)
    film_dir = library_dir / slug
    if film_dir.exists():
        import shutil

        shutil.rmtree(film_dir)
    typer.echo(f"✓ {slug} removido.")
