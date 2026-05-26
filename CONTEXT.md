# Cinemateca-imgsearch

Offline audiovisual cataloguing system for film archives. Ingests video files
and produces searchable metadata: scenes, faces, objects, natural-language
descriptions, semantic embeddings.

CLAUDE.md carries the full operational briefing and includes a legacy
"Project vocabulary" table that pre-dates this file. Terms below have been
re-grilled here against the code; the CLAUDE.md table remains the source of
truth for terms that haven't.

## Language

### Retrieval surface

**Retriever**:
The algorithm that maps a text query to a ranked list of scenes. Today the
supported values are `clip`, `hybrid` (CLIP ⊕ BM25 fused via RRF), and `bm25`.
PT user-facing label: *Modo de busca*. Image search bypasses this and always
uses image-CLIP — see flagged ambiguity below.
_Avoid_: estratégia, motor, mecanismo, engine.

**Query**:
The text or image the user submits to search the library. PT in code, docs,
and commit prose: *consulta*. PT in user-facing UI strings: *busca* — see
flagged ambiguity below.
_Avoid_: pesquisa, term.

**Search**:
The user-facing tab and the action of submitting a query. PT label: *Buscar*.
_Avoid_: pesquisar, find.

**Input type**:
The retrieval space the user submits into — `text`, `image`, `audio`, or
`fusion` in the current branch. Text uses the text retriever (`clip`, `bm25`,
or `hybrid`), image uses the image endpoint, audio uses CLAP, and fusion blends
visual and audio scores. PT label: *Tipo de entrada*.
_Avoid_: Search modality, Modalidade de busca (deprecated 2026-05-24).
Note: "Input type" is still the UI label even though `audio` and `fusion`
are true retrieval modalities; keep code comments explicit about the route
parameter name `modality`.

## Flagged ambiguities

**`busca` (PT) is overloaded.** In Portuguese strings the word does triple
duty: the tab name ("Buscar"), the noun-query the user typed, and the act
of searching. Code and design docs prefer *consulta* for the noun-query to
keep the distinction sharp; UI strings keep *busca* because it's how end
users read it.

**Retriever applies to text queries only.** The `clip`/`hybrid`/`bm25`
toggle has no effect when the user submits an image, audio, or fusion query.
Image search has its own endpoint (`/api/search/image`), audio dispatches to
CLAP, and fusion combines visual/audio scores. This is not obvious from the UI;
the popover stays visible regardless of the active input type. Either the
popover should disable when input type is not text, or the asymmetry should be
surfaced in the chip — pending decision.

**Rerank is live for text results only.** `/api/search` accepts and logs
`reranker_enabled`, then applies the cross-encoder after card enrichment so
scene descriptions are available. Image, audio, and fusion requests ignore the
text reranker.

## Example dialogue

> **Designer:** the retriever popover should sit next to the input-type chips.
>
> **Dev:** right — `Modo de busca` (the CLIP/Hybrid/BM25 toggle) lives in the
> toolrow; `Tipo de entrada` (text/image/...) is the chip row above it.
>
> **Designer:** can a user pick *Imagem* + *BM25*?
>
> **Dev:** the toggle is reachable, but image search ignores it — image
> queries always go through image-CLIP. The retriever choice only binds the
> text path.
>
> **Designer:** so the popover lies if you're in image mode.
>
> **Dev:** correct. We either disable the popover when input type isn't text,
> or document that retriever is text-only. That's a pending decision — see
> "Flagged ambiguities" above.
