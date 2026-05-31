"""Shared CLI utilities used across command families."""

from __future__ import annotations

import typer

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
)


def resolve_steps(steps_arg: str) -> set[str]:
    """Map documented short aliases (and full names) to canonical step names.

    Args:
        steps_arg: Comma-separated step tokens (short aliases or full names).

    Returns:
        Set of canonical step name strings.

    Raises:
        typer.BadParameter: When an unknown token is encountered.
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


def print_banner() -> None:
    """Print the Cinemateca ASCII banner."""
    print(
        "\n"
        "╔═══════════════════════════════════════════════════════╗\n"
        "║          Cinemateca AI  —  v0.6.0                     ║\n"
        "║  Catalogação audiovisual com IA para acervos cinema.  ║\n"
        "╚═══════════════════════════════════════════════════════╝\n",
        flush=True,
    )
