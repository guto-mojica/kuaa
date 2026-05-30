"""Scene tipo classifier + display constants.

Extracted verbatim from ``api/services/scenes_service.py`` (lines ~62–102,
645–668) during the A1 decomposition (WS-2 Task 2).
"""

from __future__ import annotations

# Allowed ``tipo`` values — paired with the ``--c-cat-<tipo>`` CSS
# variables in ``web/static/css/main.css`` that colour the scene-card
# pill background. Keep in sync with that CSS block.
_TIPOS = ("cartela", "dialogo", "exterior", "interior", "transicao")


def tipo_of(tags: list[str], description: str | None) -> str:
    """Classify a scene into one of the Mojica tipo buckets.

    The Cenas-tab scene-card pill is coloured by ``--c-cat-<tipo>``;
    this classifier picks the bucket from the LLM/manual tag list and
    (as a soft fallback) the moondream description. Order matters —
    earlier branches win when a scene has tags that match multiple
    rules:

      1. ``cartela`` — opening/closing title cards. Any of the
         ``cartela`` / ``title-card`` / ``white-writing`` tags, or the
         word "title" in the description.
      2. ``interior`` — any ``interior`` or ``baixa-luz`` tag.
      3. ``exterior`` — exact ``exterior`` tag (LLM canonical) or any
         ``rural``-prefixed tag.
      4. ``dialogo`` — two-person framing / dialogue tags.
      5. ``transicao`` — default for everything else.

    Used by ``_card_to_scene`` when assembling a ``groups_by_film``
    entry for the Cenas grid template. The values pair with the
    ``--c-cat-<tipo>`` CSS variables; new values added here MUST land
    a matching CSS variable in ``main.css`` first.
    """
    desc = (description or "").lower()
    if "title" in desc or any(
        "white-writing" in t or "cartela" in t or "title-card" in t for t in tags
    ):
        return "cartela"
    if any("interior" in t or "baixa-luz" in t for t in tags):
        return "interior"
    if "exterior" in tags or any("rural" in t for t in tags):
        return "exterior"
    if any("duas-pessoas" in t or "dialogo" in t for t in tags):
        return "dialogo"
    return "transicao"


_VALID_BUCKETS = frozenset(_TIPOS)

# Stable display order for the ``group=tipo`` headings — narrative
# arc that mirrors how a curator reads through a film (title cards →
# interiors → exteriors → dialogues → transitions). Keeps the
# headings in the same order across films instead of dict-iteration
# noise. New ``tipo`` values added in ``tipo_of`` MUST also land
# here or they'll fall through to the end in alphabetical order.
_TIPO_DISPLAY_ORDER = (
    "cartela",
    "interior",
    "exterior",
    "dialogo",
    "transicao",
)
_TIPO_LABEL = {
    "cartela": "Cartela",
    "interior": "Interior",
    "exterior": "Exterior",
    "dialogo": "Diálogo",
    "transicao": "Transição",
}
