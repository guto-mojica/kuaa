"""
cinemateca.__main__
~~~~~~~~~~~~~~~~~~~
Ponto de entrada CLI.

Uso:
    python -m cinemateca process --video data/raw/jeca_tatu.mp4
    python -m cinemateca process --video data/raw/filme.mp4 --config config/local.yaml
    python -m cinemateca process --video data/raw/filme.mp4 --steps frames,scenes
    python -m cinemateca info --video data/raw/filme.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _print_banner():
    print("""
╔═══════════════════════════════════════════════════════╗
║          Cinemateca AI  —  v0.1.0-alpha               ║
║  Catalogação audiovisual com IA para acervos cinema.  ║
╚═══════════════════════════════════════════════════════╝
""")


def cmd_process(args) -> int:
    """Executa o pipeline completo (ou etapas selecionadas)."""
    from cinemateca.config import load_config, setup_logging
    from cinemateca.pipeline import CatalogPipeline

    cfg = load_config(args.config)
    setup_logging(cfg)

    # Sobrescrever etapas se --steps fornecido
    if args.steps:
        enabled = set(args.steps.split(","))
        all_steps = [
            "frame_extraction", "scene_detection",
            "visual_analysis", "embeddings", "llm_description"
        ]
        for step in all_steps:
            setattr(cfg.pipeline.steps, step, step in enabled)

    _print_banner()
    print(f"  Vídeo  : {args.video}")
    print(f"  Config : {args.config or 'default'}")
    print(f"  Device : {cfg.hardware.device}")
    print()

    pipeline = CatalogPipeline(cfg)
    result = pipeline.run(args.video)

    print(result.summary())
    return 0 if result.success else 1


def cmd_info(args) -> int:
    """Exibe informações técnicas de um arquivo de vídeo."""
    from cinemateca.data_prep import VideoInspector

    _print_banner()
    try:
        inspector = VideoInspector(args.video)
        props = inspector.properties
        print(f"  Arquivo   : {props['filename']}")
        print(f"  Resolução : {props['width']}x{props['height']}")
        print(f"  FPS       : {props['fps']:.2f}")
        print(f"  Duração   : {props['duration_minutes']:.1f} min ({props['duration_seconds']:.1f}s)")
        print(f"  Frames    : {props['total_frames']:,}")
        print(f"  Codec     : {props['codec']}")
        print(f"  Bitrate   : {props['bit_rate_mbps']:.2f} Mbps")
        print(f"  Tamanho   : {props['file_size_gb']:.2f} GB")
        return 0
    except Exception as e:
        print(f"✗ Erro: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        prog="cinemateca",
        description="Cinemateca AI — Catalogação audiovisual com IA",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── cinemateca process ────────────────────────────────────────────────────
    p_process = subparsers.add_parser(
        "process", help="Executa o pipeline completo de catalogação"
    )
    p_process.add_argument("--video", required=True, help="Caminho do arquivo de vídeo")
    p_process.add_argument("--config", default=None, help="Caminho do arquivo config YAML")
    p_process.add_argument(
        "--steps",
        default=None,
        help="Etapas a executar, separadas por vírgula. "
             "Ex: frames,scenes,embeddings. "
             "Valores: frames,scenes,visual,embeddings,llm",
    )
    p_process.set_defaults(func=cmd_process)

    # ── cinemateca info ───────────────────────────────────────────────────────
    p_info = subparsers.add_parser(
        "info", help="Exibe informações técnicas de um vídeo"
    )
    p_info.add_argument("--video", required=True, help="Caminho do arquivo de vídeo")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
