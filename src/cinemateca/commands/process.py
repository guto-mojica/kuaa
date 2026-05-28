"""cinemateca info / process — video inspection and pipeline commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from cinemateca.commands._shared import _STEP_FULL_NAMES, print_banner, resolve_steps

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def info(
    video: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Caminho do arquivo de vídeo."
        ),
    ],
) -> None:
    """Print technical properties of a video file (resolution, fps, duration)."""
    from cinemateca.data_prep import VideoInspector

    print_banner()
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


@app.command()
def process(
    video: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Caminho do arquivo de vídeo."
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
            "audio_extract, audio_transcribe, audio_embed. "
            "Padrão: todas as etapas habilitadas na config.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(help="Caminho do arquivo config YAML (override de config/local.yaml)."),
    ] = None,
) -> None:
    """Run the full AI pipeline against a single video."""
    from cinemateca.config import load_config, setup_logging
    from cinemateca.pipeline import CatalogPipeline, slugify

    cfg = load_config(str(config) if config else None)
    setup_logging(cfg)

    if steps:
        enabled = resolve_steps(steps)
        for step in _STEP_FULL_NAMES:
            setattr(cfg.pipeline.steps, step, step in enabled)

    final_slug = slugify(slug) if slug else slugify(Path(video).stem)

    print_banner()
    print(f"  Vídeo  : {video}")
    print(f"  Slug   : {final_slug}")
    print(f"  Config : {config or 'default'}")
    print(f"  Device : {cfg.hardware.device}\n", flush=True)

    pipeline = CatalogPipeline(cfg, slug=final_slug)
    result = pipeline.run(str(video))
    print(result.summary())
    if not result.success:
        raise typer.Exit(1)
