"""cinemateca eval — eval set builder utilities."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

# Imported at module scope so tests can monkeypatch these names on the
# eval_cmd module (the `slate` command calls them via the module binding).
from cinemateca.eval.slates import ModalQuery, generate_slate, load_modal_queries

app = typer.Typer(
    name="eval",
    help="Eval set builder utilities (seed sample queries, etc.).",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Modalities the slate command can generate, plus the "all" fan-out token.
_SLATE_MODALITIES = ("text", "image", "audio", "fusion", "rhyme")


@app.command("seed")
def eval_seed(
    run: Annotated[
        str,
        typer.Option(help="Run ID (becomes <run>.queries.json in the eval root)."),
    ] = "default",
    queries: Annotated[
        int,
        typer.Option(help="Number of sample queries to write (clamped to the bundled max).", min=0),
    ] = 5,
    root: Annotated[
        Path,
        typer.Option(help="Eval data directory. Created if it doesn't exist."),
    ] = Path("data/eval"),
) -> None:
    """Write a sample queries file for the /eval grading UI."""
    from cinemateca.eval.seed import SAMPLE_QUERIES, write_seed

    path = write_seed(root=root, run_id=run, count=queries)
    written = min(max(0, queries), len(SAMPLE_QUERIES))
    typer.echo(f"✓ Wrote {written} queries to {path}")


# ─── cinemateca eval slate ───────────────────────────────────────────────────


def _slate_query_record(query: ModalQuery, rows: list[dict], *, k: int) -> dict:
    """Wrap a generated candidate slate in the /eval rows-template query contract.

    The shape matches :func:`cinemateca.eval.seed._mock_result`'s container so
    the ``/eval`` page renders generated slates exactly as it renders seeded
    ones. ``id`` is kept as the original string (``"image-01"``) — rows.html's
    header guards ``current_query.id is number`` and renders a string id
    verbatim, so no int remapping is needed; ``query_type`` is carried as an
    extra field for downstream tooling (ignored by the template).
    """
    return {
        "id": query.id,
        "query_type": query.query_type,
        "text": query.text or (f"(rhyme) {query.anchor}" if query.anchor else ""),
        "source": "slate",
        "lang": query.lang or "pt",
        "k": k,
        "candidate_count": len(rows),
        "latency_ms": None,
        "created_when": date.today().isoformat(),
        "results": rows,
    }


@app.command("slate")
def eval_slate(
    queries: Annotated[
        Path,
        typer.Option("--queries", help="m3_full-shaped query YAML to generate slates from."),
    ],
    run: Annotated[
        str,
        typer.Option("--run", help="Run ID (becomes <run>.queries.json in --root)."),
    ],
    root: Annotated[
        Path,
        typer.Option("--root", help="Eval data directory. Created if it doesn't exist."),
    ],
    modality: Annotated[
        str,
        typer.Option(
            "--modality",
            help="Modality to generate (text|image|audio|fusion|rhyme) or 'all'.",
        ),
    ] = "all",
    k: Annotated[
        int,
        typer.Option("--k", help="Candidates per query.", min=1),
    ] = 9,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Config YAML (defaults to merged default+local)."),
    ] = None,
) -> None:
    """Generate per-modality candidate slates for the /eval grading UI.

    Loads ``--queries`` (the m3_full multimodal eval set), runs the REAL
    retrieval backend for each query of the requested modality (or every
    modality when ``--modality all``), and writes
    ``<root>/<run>.queries.json`` in the rows-template contract the
    ``cinemateca eval seed`` command produces. The ``/eval`` page then
    renders the generated slates with no template changes.
    """
    from cinemateca.config import load_config

    if modality != "all" and modality not in _SLATE_MODALITIES:
        typer.echo(
            f"FAIL: --modality must be one of {('all', *_SLATE_MODALITIES)}, got {modality!r}"
        )
        raise typer.Exit(code=1)

    cfg = load_config(str(config) if config else None)
    library_dir = Path(cfg.paths.library_dir)

    all_queries = load_modal_queries(queries)
    wanted = set(_SLATE_MODALITIES) if modality == "all" else {modality}
    selected = [q for q in all_queries if q.query_type in wanted]

    records: list[dict] = []
    for q in selected:
        rows = generate_slate(query=q, cfg=cfg, library_dir=library_dir, k=k)
        records.append(_slate_query_record(q, rows, k=k))

    root.mkdir(parents=True, exist_ok=True)
    out_path = root / f"{run}.queries.json"
    out_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"✓ Wrote {len(records)} slate queries ({modality}) to {out_path}")


# ─── cinemateca eval clap-sanity ─────────────────────────────────────────────


@app.command("clap-sanity")
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
    """Run canned CLAP archival-audio queries and assert P@5 >= floor.

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
        typer.echo(f"{status}  {q['id']:25s}  P@5={p_at_5:.2f}  query={q['query']!r}")
    if any_fail:
        typer.echo(f"OVERALL: FAIL (floor={floor})")
        raise typer.Exit(code=1)
    typer.echo(f"OVERALL: PASS ({len(queries)} queries, floor={floor})")
