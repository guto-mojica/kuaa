"""
cinemateca.__main__
~~~~~~~~~~~~~~~~~~~
Ponto de entrada CLI.

Uso:
    python -m cinemateca process --video data/raw/jeca_tatu.mp4
    python -m cinemateca process --video data/raw/filme.mp4 --config config/local.yaml
    python -m cinemateca process --video data/raw/filme.mp4 --steps frames,scenes
    python -m cinemateca process --video data/raw/filme.mp4 --slug meu_filme
    python -m cinemateca info --video data/raw/filme.mp4
    python -m cinemateca reembed-all
    python -m cinemateca reembed-all --only jeca_tatu --steps embeddings,visual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_STEP_ALIASES = {
    "frames": "frame_extraction",
    "scenes": "scene_detection",
    "visual": "visual_analysis",
    "embeddings": "embeddings",
    "llm": "llm_description",
}


def _resolve_steps(steps_arg: str) -> set[str]:
    """Map documented short aliases (and full names) to canonical step names."""
    full = set(_STEP_ALIASES.values())
    out = set()
    for tok in steps_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in _STEP_ALIASES:
            out.add(_STEP_ALIASES[tok])
        elif tok in full:
            out.add(tok)
        else:
            raise ValueError(
                f"--steps: valor desconhecido {tok!r}. "
                f"Use: {','.join(_STEP_ALIASES)}"
            )
    return out


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
    from cinemateca.pipeline import CatalogPipeline, slugify

    cfg = load_config(args.config)
    setup_logging(cfg)

    # Sobrescrever etapas se --steps fornecido
    if args.steps:
        try:
            enabled = _resolve_steps(args.steps)
        except ValueError as exc:
            print(f"✗ Erro: {exc}", file=sys.stderr)
            return 1
        for step in _STEP_ALIASES.values():
            setattr(cfg.pipeline.steps, step, step in enabled)

    # Derive slug: explicit --slug overrides; otherwise slugify the video stem.
    # Always run slugify — applies to user-provided --slug too — so a hostile
    # input like "--slug ../secret" can't escape the library root.
    slug = slugify(args.slug) if args.slug else slugify(Path(args.video).stem)

    _print_banner()
    print(f"  Vídeo  : {args.video}")
    print(f"  Slug   : {slug}")
    print(f"  Config : {args.config or 'default'}")
    print(f"  Device : {cfg.hardware.device}")
    print()

    pipeline = CatalogPipeline(cfg, slug=slug)
    result = pipeline.run(args.video)

    print(result.summary())
    return 0 if result.success else 1


def cmd_reembed_all(args) -> int:
    """Re-run the embeddings step (or other steps) across every registered film.

    Walks ``data/library/films.json`` and, for each film, drives the
    existing :class:`CatalogPipeline` with the **registered** slug —
    avoiding the filename-to-slug drift that bites the bare
    ``process --video <path>`` form (a renamed file or a special-char
    title computes a different slug than the one already in the
    registry, the pipeline registers a NEW empty film, and every step
    skips).

    By default it clears the per-film ``.npy`` and ``index_mapping.json``
    before running because the pipeline's ``skip_existing`` would
    otherwise silently no-op the embeddings step when an old index
    file is present. Pass ``--keep-existing`` to opt out (useful when
    you only want to extend missing artifacts, not rebuild them).
    """
    from cinemateca.config import load_config, setup_logging
    from cinemateca.library import scan_library
    from cinemateca.pipeline import CatalogPipeline

    cfg = load_config(args.config)
    setup_logging(cfg)

    try:
        enabled = _resolve_steps(args.steps)
    except ValueError as exc:
        print(f"✗ Erro: {exc}", file=sys.stderr)
        return 1
    for step in _STEP_ALIASES.values():
        setattr(cfg.pipeline.steps, step, step in enabled)

    library_dir = Path(cfg.paths.library_dir)
    films = scan_library(library_dir)
    if not films:
        print(
            f"✗ Nenhum filme registrado em {library_dir/'films.json'}",
            file=sys.stderr,
        )
        return 1

    only = set(args.only or [])
    if only:
        unknown = only - {f.slug for f in films}
        if unknown:
            print(
                f"✗ Slugs não registrados: {', '.join(sorted(unknown))}",
                file=sys.stderr,
            )
            return 1
        films = [f for f in films if f.slug in only]

    # flush=True throughout — pipeline.run() uses logger.info() which
    # writes to stderr (line-buffered); our stdout prints would otherwise
    # appear AFTER the pipeline's logs in a combined 2>&1 capture even
    # though they execute first.
    _print_banner()
    sys.stdout.flush()
    print(f"  Filmes a reprocessar : {len(films)}", flush=True)
    print(f"  Etapas               : {','.join(sorted(enabled))}", flush=True)
    print(f"  Config               : {args.config or 'default'}", flush=True)
    print(f"  Apaga .npy antes     : {not args.keep_existing}", flush=True)
    print(flush=True)

    raw_dir = Path(cfg.paths.raw_dir)
    summary: list[tuple[str, str, float]] = []
    for film in films:
        # Per-film raw layout first; fall back to the legacy data/raw/
        # mirror so Train-Robbery-style films (registered without a
        # per-film raw/) still rebuild correctly.
        candidates = [film.raw_path, raw_dir / film.raw_path.name]
        video = next((p for p in candidates if p.exists()), None)
        if video is None:
            tried = ", ".join(str(p) for p in candidates)
            print(f"⏭  {film.slug} — raw não encontrado ({tried})\n", flush=True)
            summary.append((film.slug, "skipped (no raw)", 0.0))
            continue

        if not args.keep_existing and "embeddings" in enabled:
            emb_dir = library_dir / film.slug / "embeddings"
            for fname in ("keyframe_embeddings.npy", "index_mapping.json"):
                p = emb_dir / fname
                if p.exists():
                    p.unlink()

        print(f"━━━ {film.slug} ━━━", flush=True)
        pipeline = CatalogPipeline(cfg, slug=film.slug)
        result = pipeline.run(str(video))
        status = "OK" if result.success else "FAIL"
        elapsed = float(getattr(result, "total_duration_s", 0.0))
        summary.append((film.slug, status, elapsed))
        print(flush=True)

    print("━" * 60, flush=True)
    print(f"  {'STATUS':<18}  {'TIME':>8}  SLUG", flush=True)
    for slug, status, elapsed in summary:
        print(f"  {status:<18}  {elapsed:>6.1f}s  {slug}", flush=True)
    n_ok = sum(1 for _, s, _ in summary if s == "OK")
    print(f"\n  {n_ok}/{len(summary)} success", flush=True)
    return 0 if n_ok == len(summary) else 1


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
    p_process.add_argument(
        "--slug",
        default=None,
        help="Identificador do filme na biblioteca (ex: jeca_tatu). "
             "Padrão: stem do nome do vídeo, slugificado. "
             "Saída em data/library/<slug>/.",
    )
    p_process.set_defaults(func=cmd_process)

    # ── cinemateca reembed-all ────────────────────────────────────────────────
    p_re = subparsers.add_parser(
        "reembed-all",
        help="Reconstrói embeddings (ou outras etapas) para todos os filmes "
             "registrados em data/library/films.json",
    )
    p_re.add_argument(
        "--config", default=None, help="Caminho do arquivo config YAML",
    )
    p_re.add_argument(
        "--steps", default="embeddings",
        help="Etapas a executar, separadas por vírgula. "
             "Default: embeddings. "
             "Valores: frames,scenes,visual,embeddings,llm",
    )
    p_re.add_argument(
        "--only", action="append", default=None,
        help="Slug a processar (repetível). Default: todos os filmes registrados. "
             "Ex: --only jeca_tatu --only edwin_porter-the_great_train_robbery_1903",
    )
    p_re.add_argument(
        "--keep-existing", action="store_true",
        help="Não apaga .npy / index_mapping.json antes de re-rodar. "
             "Default: apaga, evitando que skip_existing pule a etapa.",
    )
    p_re.set_defaults(func=cmd_reembed_all)

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
