"""kuaa motion — compute per-scene motion statistics for indexed films.

A separate command rather than a pipeline step, because it is purely
additive: it reads ``scene_cuts.json`` and the source video, writes
``scene_motion.json``, and invalidates nothing. Films without it keep
working exactly as before.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from kuaa.commands._shared import print_banner

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

logger = logging.getLogger(__name__)


@app.command("run")
def run(
    slug: Annotated[
        str | None,
        typer.Option("--only", help="Processar apenas este filme (slug). Padrão: todos."),
    ] = None,
    sample_fps: Annotated[
        float | None,
        typer.Option(help="Quadros por segundo amostrados. Padrão: motion.sample_fps."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Recalcular mesmo se scene_motion.json já existir."),
    ] = False,
) -> None:
    """Compute optical-flow motion statistics per scene.

    CPU-only, one decode pass per film at ``--sample-fps``. Writes
    ``metadata/scene_motion.json``; existing artefacts are untouched.
    """
    from kuaa.config import load_config
    from kuaa.library import load_registry
    from kuaa.motion import analyze_video, save_motion
    from kuaa.motion.io import MOTION_FILENAME
    from kuaa.scene_detector import read_cutset

    print_banner()
    cfg = load_config()
    fps_sample = cfg.motion.sample_fps if sample_fps is None else sample_fps
    library_dir = Path(cfg.paths.library_dir)
    raw_dir = Path(cfg.paths.raw_dir)

    registry = load_registry(library_dir)
    if slug:
        if slug not in registry:
            typer.secho(f"Filme não encontrado: {slug}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        registry = {slug: registry[slug]}

    processed = 0
    for film_slug, film in sorted(registry.items()):
        metadata_dir = library_dir / film_slug / "metadata"
        out_path = metadata_dir / MOTION_FILENAME

        if out_path.exists() and not overwrite:
            typer.echo(f"  {film_slug}: já processado (use --overwrite)")
            continue

        cutset = read_cutset(metadata_dir / "scene_cuts.json")
        if cutset is None:
            typer.secho(
                f"  {film_slug}: sem scene_cuts.json — rode a detecção de cenas antes",
                fg=typer.colors.YELLOW,
            )
            continue

        video_path = _resolve_video(film, raw_dir)
        if video_path is None:
            typer.secho(f"  {film_slug}: vídeo de origem não encontrado", fg=typer.colors.YELLOW)
            continue

        boundaries = cutset.scene_boundaries()
        typer.echo(f"  {film_slug}: {len(boundaries)} cenas — analisando fluxo óptico…")
        try:
            records = analyze_video(
                video_path,
                boundaries,
                cutset.fps,
                sample_fps=fps_sample,
                max_pairs_per_scene=cfg.motion.max_pairs_per_scene,
            )
        except OSError as exc:
            typer.secho(f"  {film_slug}: falha ao ler o vídeo ({exc})", fg=typer.colors.RED)
            continue

        save_motion(metadata_dir, records)
        measured = sum(1 for r in records if r.sampled_pairs)
        static = sum(1 for r in records if r.camera_motion == "static")
        moving = sum(1 for r in records if r.subject_motion == "moving")
        typer.secho(
            f"  {film_slug}: {measured}/{len(records)} cenas medidas — "
            f"{static} estáticas, {moving} com movimento de cena",
            fg=typer.colors.GREEN,
        )
        processed += 1

    typer.echo(f"\nConcluído: {processed} filme(s) processado(s).")


def _resolve_video(film: dict, raw_dir: Path) -> Path | None:
    """Locate a film's source video from its registry entry."""
    for key in ("raw_path", "source_path", "video_path"):
        value = film.get(key)
        if value and Path(value).exists():
            return Path(value)
    raw_filename = film.get("raw_filename")
    if raw_filename:
        candidate = raw_dir / str(raw_filename)
        if candidate.exists():
            return candidate
    return None
