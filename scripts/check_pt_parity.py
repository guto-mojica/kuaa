#!/usr/bin/env python
"""Assert Portuguese queries actually retrieve, and report PT↔EN agreement.

Why this exists
---------------
The describer prompts are English, so the BM25 corpus is English while the
interface, the curators, and the queries are Brazilian Portuguese. Before
the bilingual index-time expansion landed, PT content queries returned
*literally zero* BM25 hits on ``jeca_tatu_1959`` while their English
equivalents returned full result sets — a failure no test caught, because
nothing asserted on retrieval quality in either language.

This is the regression guard for that. It fails loudly if any PT query in
the pair set goes back to scoring nothing.

Two numbers are reported per pair:

* **hits** — how many results each side returns. The hard gate; PT must be
  non-zero.
* **overlap@k** — how many scene ids the PT and EN phrasings agree on.
  Informational, not gated: PT and EN expand to different token sets with
  different IDF profiles, so perfect agreement is neither expected nor
  desirable. For reference, the SigLIP2 dense leg — which is natively
  multilingual and needs no lexicon — agrees with itself across languages
  at roughly 4–7 of 10.

Usage::

    uv run python scripts/check_pt_parity.py
    uv run python scripts/check_pt_parity.py --slug jeca_tatu_1959 --k 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kuaa.config import load_config
from kuaa.search.bm25 import bm25_index_for_dir, resolve_bm25_kwargs

logger = logging.getLogger(__name__)

# PT query paired with the English phrasing a curator would otherwise have
# had to type. Content words only — these are the terms an archive actually
# gets asked for.
QUERY_PAIRS: tuple[tuple[str, str], ...] = (
    ("homem a cavalo", "man on a horse"),
    ("mulher de vestido", "woman in a dress"),
    ("dois homens conversando", "two men talking"),
    ("trabalhador rural arando o campo", "rural worker plowing the field"),
    ("casa humilde de sapé", "simple thatched hut"),
    ("carro de boi", "ox cart"),
    ("cavalo", "horse"),
    ("chapéu de palha", "straw hat"),
    ("cena noturna", "night scene"),
    ("crianças brincando", "children playing"),
    ("cerca de madeira", "wooden fence"),
    ("árvores ao fundo", "trees in the background"),
)


def main() -> int:
    """Return 0 when every PT query retrieves, 1 otherwise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="jeca_tatu_1959", help="film slug to probe")
    parser.add_argument("--k", type=int, default=10, help="results per query")
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="override the configured library dir",
    )
    args = parser.parse_args()

    cfg = load_config()
    library_dir = args.library_dir or Path(cfg.paths.library_dir)
    metadata_dir = library_dir / args.slug / "metadata"
    if not metadata_dir.is_dir():
        print(f"FAIL: no metadata dir at {metadata_dir}", file=sys.stderr)
        return 1

    index = bm25_index_for_dir(metadata_dir=metadata_dir, **resolve_bm25_kwargs(cfg))
    if not index.scene_ids:
        print(f"FAIL: empty BM25 index for {args.slug}", file=sys.stderr)
        return 1

    print(f"BM25 PT parity — {args.slug} ({len(index.scene_ids)} docs, k={args.k})\n")
    print(f"  {'PT query':34s} {'PT':>4s} {'EN':>4s} {'overlap':>8s}")
    print(f"  {'-' * 34} {'-' * 4} {'-' * 4} {'-' * 8}")

    dead: list[str] = []
    for pt, en in QUERY_PAIRS:
        pt_hits = index.query(pt, args.k)
        en_hits = index.query(en, args.k)
        overlap = len({s for s, _ in pt_hits} & {s for s, _ in en_hits})
        flag = "" if pt_hits else "  <-- NO PT HITS"
        if not pt_hits:
            dead.append(pt)
        print(f"  {pt:34s} {len(pt_hits):4d} {len(en_hits):4d} {overlap:6d}/{args.k}{flag}")

    print()
    if dead:
        print(f"FAIL: {len(dead)}/{len(QUERY_PAIRS)} PT queries returned nothing:", file=sys.stderr)
        for q in dead:
            print(f"  - {q}", file=sys.stderr)
        print(
            "\nCheck `search.bm25.bilingual` and `search.bm25.tokenizer` in config.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: all {len(QUERY_PAIRS)} PT queries retrieve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
