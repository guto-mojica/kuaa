"""
cinemateca.__main__
~~~~~~~~~~~~~~~~~~~
Unified CLI entry point (Typer).

The whole tool — both the AI pipeline and the FastAPI web app — is
driven from a single ``cinemateca`` command so options stay
discoverable via ``--help`` at every level. There is no separate
``app.py`` invocation to remember.

Tree:
    cinemateca serve                  # FastAPI web app (was: uv run app.py)
    cinemateca info VIDEO             # Video properties
    cinemateca process VIDEO ...      # Full pipeline (single film)
    cinemateca library list           # Show registered films + state
    cinemateca library reembed ...    # Rebuild embeddings across registry
    cinemateca library delete SLUG    # Remove a film + artifacts
    cinemateca config show            # Dump effective merged config
    cinemateca eval seed ...          # Seed sample queries for /eval

Each subcommand has its own ``--help``. ``cinemateca`` alone prints
the command tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="cinemateca",
    help="Cinemateca AI — offline audiovisual cataloguing for film archives.",
    no_args_is_help=True,
    add_completion=False,  # We don't ship completion install commands — users wire it themselves.
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

library_app = typer.Typer(
    name="library",
    help="Operations across the registered film library " "(data/library/films.json).",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
config_app = typer.Typer(
    name="config",
    help="Inspect the merged effective configuration " "(default.yaml ⊕ local.yaml).",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
eval_app = typer.Typer(
    name="eval",
    help="Eval set builder utilities (seed sample queries, etc.).",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(library_app, name="library")
app.add_typer(config_app, name="config")
app.add_typer(eval_app, name="eval")


# ── Shared option / step resolution ───────────────────────────────────────────

_STEP_ALIASES: dict[str, str] = {
    "frames": "frame_extraction",
    "scenes": "scene_detection",
    "visual": "visual_analysis",
    "embeddings": "embeddings",
    "llm": "llm_description",
}
_STEP_FULL_NAMES: tuple[str, ...] = (
    "frame_extraction",
    "scene_detection",
    "visual_analysis",
    "embeddings",
    "llm_description",
    "audio_extract",
    "audio_embed",
)


def _resolve_steps(steps_arg: str) -> set[str]:
    """Map documented short aliases (and full names) to canonical step names.

    Raises ``typer.BadParameter`` with a clear message on unknown tokens so
    the CLI surfaces the error inline instead of crashing with a stack trace.
    """
    full = set(_STEP_FULL_NAMES)
    out: set[str] = set()
    for raw in steps_arg.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok in _STEP_ALIASES:
            out.add(_STEP_ALIASES[tok])
        elif tok in full:
            out.add(tok)
        else:
            known = ",".join(list(_STEP_ALIASES) + list(_STEP_FULL_NAMES))
            raise typer.BadParameter(
                f"valor desconhecido {tok!r}. Use: {known}",
                param_hint="--steps",
            )
    return out


def _print_banner() -> None:
    print(
        "\n"
        "╔═══════════════════════════════════════════════════════╗\n"
        "║          Cinemateca AI  —  v0.1.0-alpha               ║\n"
        "║  Catalogação audiovisual com IA para acervos cinema.  ║\n"
        "╚═══════════════════════════════════════════════════════╝\n",
        flush=True,
    )


# ─── cinemateca serve ────────────────────────────────────────────────────────


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8501,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload/--no-reload",
            help="Auto-reload on code changes (dev mode).",
        ),
    ] = True,
) -> None:
    """Run the FastAPI web app (the v0.3+ UI).

    Replaces the legacy ``uv run app.py`` invocation. Opens
    ``http://<host>:<port>``.
    """
    import uvicorn
    from pathlib import Path

    project_root = str(Path(__file__).parent.parent.parent)
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        app_dir=project_root,
    )


# ─── cinemateca info ─────────────────────────────────────────────────────────


@app.command()
def info(
    video: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Caminho do arquivo de vídeo.",
        ),
    ],
) -> None:
    """Print technical properties of a video file (resolution, fps, duration)."""
    from cinemateca.data_prep import VideoInspector

    _print_banner()
    try:
        props = VideoInspector(str(video)).properties
        print(f"  Arquivo   : {props['filename']}")
        print(f"  Resolução : {props['width']}x{props['height']}")
        print(f"  FPS       : {props['fps']:.2f}")
        print(
            f"  Duração   : {props['duration_minutes']:.1f} min "
            f"({props['duration_seconds']:.1f}s)"
        )
        print(f"  Frames    : {props['total_frames']:,}")
        print(f"  Codec     : {props['codec']}")
        print(f"  Bitrate   : {props['bit_rate_mbps']:.2f} Mbps")
        print(f"  Tamanho   : {props['file_size_gb']:.2f} GB")
    except Exception as exc:
        typer.echo(f"✗ Erro: {exc}", err=True)
        raise typer.Exit(1) from exc


# ─── cinemateca process ──────────────────────────────────────────────────────


@app.command()
def process(
    video: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Caminho do arquivo de vídeo.",
        ),
    ],
    slug: Annotated[
        str | None,
        typer.Option(
            help="Identificador do filme na biblioteca (ex: jeca_tatu). "
            "Padrão: stem do nome do vídeo, slugificado. "
            "Saída em data/library/<slug>/.",
        ),
    ] = None,
    steps: Annotated[
        str | None,
        typer.Option(
            help="Etapas a executar, separadas por vírgula. "
            "Valores: frames, scenes, visual, embeddings, llm, "
            "audio_extract, audio_embed. "
            "Padrão: todas as etapas habilitadas na config.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            help="Caminho do arquivo config YAML (override de config/local.yaml).",
        ),
    ] = None,
) -> None:
    """Run the full AI pipeline against a single video.

    Use ``cinemateca library reembed`` if you want to re-run the same
    steps across every registered film — that variant looks up the
    registered slug per film and avoids filename→slug drift.
    """
    from cinemateca.config import load_config, setup_logging
    from cinemateca.pipeline import CatalogPipeline, slugify

    cfg = load_config(str(config) if config else None)
    setup_logging(cfg)

    if steps:
        enabled = _resolve_steps(steps)
        for step in _STEP_FULL_NAMES:
            setattr(cfg.pipeline.steps, step, step in enabled)

    # Always slugify — applies to user-provided --slug too — so a hostile
    # input like "--slug ../secret" can't escape the library root.
    final_slug = slugify(slug) if slug else slugify(Path(video).stem)

    _print_banner()
    print(f"  Vídeo  : {video}")
    print(f"  Slug   : {final_slug}")
    print(f"  Config : {config or 'default'}")
    print(f"  Device : {cfg.hardware.device}\n", flush=True)

    pipeline = CatalogPipeline(cfg, slug=final_slug)
    result = pipeline.run(str(video))
    print(result.summary())
    if not result.success:
        raise typer.Exit(1)


# ─── cinemateca library list ─────────────────────────────────────────────────


@library_app.command("list")
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
        print(
            f"Nenhum filme registrado em " f"{Path(cfg.paths.library_dir)/'films.json'}",
        )
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


# ─── cinemateca library reembed ──────────────────────────────────────────────


@library_app.command("reembed")
def library_reembed(
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Slug a processar (repetível). Padrão: todos os filmes registrados.",
        ),
    ] = None,
    steps: Annotated[
        str,
        typer.Option(
            help="Etapas a executar, separadas por vírgula. "
            "Valores: frames, scenes, visual, embeddings, llm, "
            "audio_extract, audio_embed.",
        ),
    ] = "embeddings",
    keep_existing: Annotated[
        bool,
        typer.Option(
            "--keep-existing",
            help="Não apaga .npy / index_mapping.json antes de re-rodar. "
            "Padrão: apaga, evitando que skip_existing pule a etapa.",
        ),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option(help="Caminho do arquivo config YAML."),
    ] = None,
) -> None:
    """Rebuild artifacts across every registered film (or a subset via ``--only``).

    Drives the pipeline with the **registered** slug from films.json,
    avoiding the filename→slug drift that bites the bare ``process``
    form when a filename slugifies to something different from its
    registered slug. By default, clears the stale ``.npy`` and
    ``index_mapping.json`` before each run because the pipeline's
    ``skip_existing`` would otherwise silently no-op the embeddings step.
    """
    from cinemateca.config import load_config, setup_logging
    from cinemateca.library import scan_library
    from cinemateca.pipeline import CatalogPipeline

    cfg = load_config(str(config) if config else None)
    setup_logging(cfg)

    enabled = _resolve_steps(steps)
    for step in _STEP_FULL_NAMES:
        setattr(cfg.pipeline.steps, step, step in enabled)

    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    if not films:
        typer.echo(
            f"✗ Nenhum filme registrado em {library_dir/'films.json'}",
            err=True,
        )
        raise typer.Exit(1)

    only_set = set(only or [])
    if only_set:
        unknown = only_set - {f.slug for f in films}
        if unknown:
            typer.echo(
                f"✗ Slugs não registrados: {', '.join(sorted(unknown))}",
                err=True,
            )
            raise typer.Exit(1)
        films = [f for f in films if f.slug in only_set]

    _print_banner()
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
            if "audio_embed" in enabled:
                stale.append(("audio", ("clap_embeddings.npy", "audio_mapping.json")))
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


# ─── cinemateca library delete ───────────────────────────────────────────────


@library_app.command("delete")
def library_delete(
    slug: Annotated[
        str,
        typer.Argument(help="Slug do filme a remover (ex: jeca_tatu)."),
    ],
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Confirma sem prompt interativo (use em scripts).",
        ),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option(help="Caminho do arquivo config YAML."),
    ] = None,
) -> None:
    """Remove a film from the registry (and delete its on-disk artifacts).

    Destructive — requires explicit confirmation unless ``--yes`` is passed.
    """
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
    # ``delete_film`` removes the registry entry but may leave the on-disk
    # directory; clear it ourselves so "delete" really means "delete".
    film_dir = library_dir / slug
    if film_dir.exists():
        import shutil

        shutil.rmtree(film_dir)
    typer.echo(f"✓ {slug} removido.")


# ─── cinemateca config show ──────────────────────────────────────────────────


@config_app.command("show")
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


# ─── cinemateca eval seed ────────────────────────────────────────────────────

@eval_app.command("seed")
def eval_seed(
    run: Annotated[
        str,
        typer.Option(help="Run ID (becomes <run>.queries.json in the eval root)."),
    ] = "default",
    queries: Annotated[
        int,
        typer.Option(
            help="Number of sample queries to write (clamped to the bundled max).",
            min=0,
        ),
    ] = 5,
    root: Annotated[
        Path,
        typer.Option(
            help="Eval data directory. Created if it doesn't exist.",
        ),
    ] = Path("data/eval"),
) -> None:
    """Write a sample queries file for the /eval grading UI.

    The seed slate is hand-crafted placeholder data (five queries, nine
    candidates each) used to populate the standalone grading workspace
    on a fresh install. The Month 3 curator-annotation work will replace
    these placeholders with real /api/search results once the
    multi-film library is migrated.
    """
    from cinemateca.eval.seed import SAMPLE_QUERIES, write_seed

    path = write_seed(root=root, run_id=run, count=queries)
    written = min(max(0, queries), len(SAMPLE_QUERIES))
    typer.echo(f"✓ Wrote {written} queries to {path}")


# ─── cinemateca eval clap-sanity ─────────────────────────────────────────────


@eval_app.command("clap-sanity")
def eval_clap_sanity(
    fixture: Annotated[
        Path,
        typer.Option(
            "--fixture",
            "-f",
            help="Path to the clap_sanity_queries.json fixture.",
        ),
    ] = Path("tests/fixtures/clap_sanity_queries.json"),
    library: Annotated[
        str | None,
        typer.Option(help="Override library slug (defaults to fixture value)."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(help="Caminho do arquivo config YAML."),
    ] = None,
) -> None:
    """Run canned CLAP archival-audio queries and assert P@5 ≥ floor.

    Each query in the fixture declares ``expected_scene_ids`` — the scene
    ids any healthy CLAP backend must surface in the top-5 result list.
    The command computes P@5 per query, prints a one-line status, and
    exits non-zero if any query falls below the fixture's
    ``p_at_5_floor`` (default 0.4). Designed to gate pre-commit /
    CI runs against silent CLAP-backend regressions.
    """
    import json

    from cinemateca.config import load_config
    from cinemateca.library.context import FilmContext
    from cinemateca.models.registry import get_audio_embedder
    from cinemateca.search.audio import load_audio_index, search_audio

    data = json.loads(fixture.read_text(encoding="utf-8"))
    slug = library or data.get("library")
    if not slug:
        typer.echo("FAIL: fixture missing 'library' and no --library override given")
        raise typer.Exit(code=1)
    floor = float(data.get("p_at_5_floor", 0.4))
    cfg = load_config(str(config) if config else None)
    ctx = FilmContext.for_film(cfg, slug)
    # Per-film audio dir lives next to metadata under <library_dir>/<slug>/audio.
    audio_dir = Path(ctx.metadata_dir).parent / "audio"
    idx = load_audio_index(audio_dir)
    if idx is None:
        typer.echo(f"FAIL: no CLAP index at {audio_dir}")
        raise typer.Exit(code=1)
    embedder = get_audio_embedder(cfg, device=None)
    any_fail = False
    queries = data.get("queries", []) or []
    for q in queries:
        hits = search_audio(idx, embedder, q["query"], top_k=5)
        hit_ids = {int(h["scene_id"]) for h in hits}
        expected = {int(s) for s in q.get("expected_scene_ids", [])}
        # P@5 is "of the top-5 returned, how many were in the expected set",
        # divided by 5 (not by len(expected)) so the floor is comparable
        # across queries with different expected-set sizes.
        p_at_5 = len(hit_ids & expected) / 5.0
        status = "PASS" if p_at_5 >= floor else "FAIL"
        if status == "FAIL":
            any_fail = True
        typer.echo(
            f"{status}  {q['id']:25s}  P@5={p_at_5:.2f}  query={q['query']!r}"
        )
    if any_fail:
        typer.echo(f"OVERALL: FAIL (floor={floor})")
        raise typer.Exit(code=1)
    typer.echo(f"OVERALL: PASS ({len(queries)} queries, floor={floor})")


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    """Console-script entry (``[project.scripts]`` → ``cinemateca``)."""
    app()


if __name__ == "__main__":
    main()
