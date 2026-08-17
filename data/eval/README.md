# Evaluation Data

Small, reviewable evaluation datasets. Files here are source data and should
stay light enough to read in a pull request.

## Two evaluation systems, not one

This is the distinction the rest of this document depends on, and nothing in
the filenames reveals it. There are two ways to get a number out of this repo.
They produce different kinds of number and they fail differently.

| | **Grading** | **Automatic scoring** |
|---|---|---|
| Where labels come from | A human, one keypress per scene, after seeing the scene | The `relevant_scene_ids` / `relevance` fields in a query YAML, guessed in advance |
| Entry point | `kuaa eval slate`, then the `/eval` pane | `scripts/run_eval.py`, `scripts/run_ablation.py` |
| Output | `<run>.jsonl` | `docs/EVALUATION_RESULTS.md` |
| Honesty tier | Ground truth | **HY** — hypothesis (see `kuaa.eval.proxy`) |

**Grading is the one that produces ground truth.** Automatic scoring exists so
a table can be published with zero human grades; `kuaa.eval.proxy` is explicit
that its labels are "pre-curator hypotheses... never ground truth."

### How automatic scoring fails quietly

A label is matched to a retrieved scene by id. In `dcg_at_k`
(`src/kuaa/eval/metrics.py`):

```python
grade = grades.get(sid, 0.0)
```

A retrieved scene absent from the label dict scores zero. Not an error — zero.
So when a query set's labels name scenes from a corpus that no longer matches
the library, every retrieved scene scores zero, nDCG collapses toward the
floor, and a number still prints. It reads as a measurement of bad retrieval.
It is a measurement of nothing.

There is a detector and it is not wired to stop anything: `evaluate_query`
fills `missing_relevant_scene_ids`, and `run_retrieval_eval` appends
`"relevant scene ids missing from index"` to a `warnings` list in the report.
The run completes and publishes either way.

Two consequences, both load-bearing:

- Hypothesis labels are only as good as the corpus they were written against,
  and they degrade **invisibly** when it changes.
- This is why `corpus01_queries.yaml` carries no labels at all, and why the
  pre-corpus01 sets were deleted rather than repointed.

---

## Grading

### Three files, three roles

The names do not make the distinction obvious, so:

| File | Role | Written by |
|---|---|---|
| `<name>.yaml` | The **queries** | A human, by hand |
| `<run>.queries.json` | The **pooled candidates** to judge | `kuaa eval slate` |
| `<run>.jsonl` | The **judgments** | The `/eval` pane, one row per keypress |

The `.jsonl` is the ground truth and outlives everything else. Any retriever —
including ones that do not exist yet — can be scored against it, which is the
whole reason the candidates are pooled rather than taken from one retriever.

### The full sequence

It is not a single command. Steps 1 and 3 are human work; 2 and 4 are not.

**1. Author the queries by hand.** A `.yaml` in this directory, no labels. This
is editorial judgment about what the archive should be asked, and it is the
step nothing can do for you. See *Authoring a query set* below.

**2. Generate the pooled candidates.**

```bash
uv run kuaa eval slate --queries data/eval/corpus01_queries.yaml \
  --run corpus01 --root data/eval --modality all --k 11
```

This calls the real retrievers and writes `corpus01.queries.json`. No judgment
in it.

**`--k` must be at least the number of films in the library.** The cross-film
merge interleaves round-robin by sorted slug — rank 1 from every film, then
rank 2 — so a `k` below the film count cuts mid-rotation and the
last-sorting films are excluded from every query in the run, alphabetically
rather than by relevance. `corpus01` has 11 films, hence `--k 11`. The
composition report checks this and exits non-zero if any searched film never
reached a pool, so the constraint is enforced rather than remembered.

It also writes two things beside the pool:

- **`corpus01.pool_composition.json`** — which retriever contributed what.
  The command exits non-zero when the pool is not fit to grade: a source
  retriever that widened it by nothing, two variants that are not
  distinguishable, a film that never appeared. Read it before booking anyone's
  afternoon.
- **`scene_manifests`**, inside the pool file — a hash of each film's scene
  numbering at generation time. `scene_id` is an ordinal that a cut edit
  renumbers, so without this a grade collected today silently points at a
  different scene tomorrow. `/eval` refuses to render (409) when a film's
  numbering has moved since generation; regenerate the pool.

**3. Point the pane at the run, then grade.** The `/eval` pane reads
`cfg.eval.run_id`, which defaults to `default` — passing `--run corpus01` above
does **not** move it. Set it in `config/local.yaml`:

```yaml
eval:
  root: data/eval
  run_id: corpus01
```

Then:

```bash
EVAL_ADMIN_TOKEN=dev uv run kuaa serve     # then open /eval?token=dev
```

The pane is admin-gated and unlinked from the nav — it is invisible unless
`EVAL_ADMIN_TOKEN` is set. It shows one query at a time: keyframe thumbnails,
grade buttons, a progress bar, and live inter-annotator agreement.

**Set a `grader` cookie to your name before grading.** It is the field the
inter-annotator κ reads to tell two graders apart, and it defaults to `anon`.
The pane resumes at your first ungraded row, per grader.

**4. The `.jsonl` accumulates.** One row per keypress, appended.

---

## The grading contract — v1, frozen

Everything in this section is cheap to decide now and expensive to change
after the first keypress. Changing the scale orphans every grade already
collected; deciding the adjudication rule after the first real disagreement
means deciding it about that disagreement; stating the κ bar after seeing κ
means stating whatever number came out. So all three are written down first,
and this section is versioned.

**Schema version: `1`.** Rows in a `.jsonl` written under this contract are
comparable to each other and to nothing else. Bump it and you have started a
new dataset, not extended this one.

### 1. Label scale — graded 0–3

| Key | Value | Meaning for THIS archive |
|---|---|---|
| `0` | IRRELEVANT | The query's subject is not in this scene. A viewer looking for it would say no. |
| `1` | WEAKLY | Present but incidental — in the background, in one corner, or only by association. Would not be the scene you cite. |
| `2` | RELEVANT | The query's subject is genuinely in the scene and legible. A reasonable answer. |
| `3` | HIGHLY_RELEVANT | The scene is *about* the query's subject. The one you would pull for a curator who asked. |
| `S` | SKIP (`-1`) | Explicit "no opinion" — the keyframe is too dark, too damaged, or too ambiguous to judge. **Not** "I'll come back to it", and distinct from ungraded. |

Graded rather than binary, and the reason is mechanical: `grader_metrics._gain`
computes `2^g - 1`, so the scale is already assumed to be integer-graded
everywhere the metrics are computed. Collapsing it to binary later is a
lossless projection; expanding binary labels to graded ones later is not
possible without re-grading. Grade at the finer resolution.

**Judge the scene, not the keyframe.** The thumbnail is a representative
frame, not the unit of judgment. Where the frame is ambiguous but the
description and timecode make the scene clear, grade the scene.

**Judge against the query, not against the system.** A pool deliberately
contains candidates every retriever proposed, including ones that are plainly
wrong. `0` is a normal, frequent, useful answer.

### 2. Adjudication protocol

1. Two graders overlap on a declared subset of queries (see below).
2. A **disagreement** is |Δ| ≥ 2 on the same `(query, scene)` — adjacent
   grades (`1` vs `2`) are scale noise, not conflict. This is the same
   threshold the pane's compare-mode chip already tints red.
3. Every disagreement goes to a **consensus pass**: both graders look at the
   scene together and one of them re-grades. The append-only log keeps both
   original votes, so κ is still computed on the *independent* first pass —
   consensus must not be allowed to inflate the agreement figure it is
   supposed to be measured by.
4. If consensus is not reached, the scene is graded `S`. It leaves the scored
   set rather than being resolved by seniority.
5. `SKIP` is excluded from κ and from every metric. It is not a grade.

### 3. Pre-registered κ threshold

Stated before any κ is computed, so it cannot be retrofitted to the number
that comes out:

| Cohen's κ | Reading |
|---|---|
| **≥ 0.60** | Labels are usable as ground truth; publish retrieval numbers from them. |
| **0.40 – 0.59** | Publish only with the κ stated beside every number, and treat differences smaller than the between-grader spread as noise. |
| **< 0.40** | The labels measure the graders, not the retrieval. Do not publish. Fix the grade definitions above and re-grade. |

**The overlap subset is ~10 queries, and that is a directional read, not a
precise one.** κ on ~10 queries × ~30 candidates carries a wide confidence
interval; it can tell 0.2 from 0.7 and cannot tell 0.55 from 0.65. Report it
as a range-check against the table above, never as a point estimate.

**Intra-grader consistency, free.** The same scene appears in the pools of
different queries, so each grader can be compared against themselves without
labelling anything extra. Divergence there is the cheapest early warning that
a grader was fatiguing or drifting mid-session — check it before reading κ,
because a grader who disagrees with themselves explains a low κ that would
otherwise be read as disagreement with the other person.

---

### The grades

The scale is defined above. Summarised for the pane:

| Key | Value | Meaning |
|---|---|---|
| `0` | IRRELEVANT | Definitely not a match |
| `1` | WEAKLY | Tangentially related |
| `2` | RELEVANT | A reasonable match |
| `3` | HIGHLY_RELEVANT | Exemplary match |
| `S` | SKIP (`-1`) | Explicit "no opinion" — distinct from ungraded |

### Grade pools, never slates

Candidates put in front of a grader come from the union of every retriever
under comparison (`kuaa.eval.registry.RETRIEVER_REGISTRY`), never from a single
`mode`. Judgments drawn from one retriever's output are valid only for that
retriever, and silently penalise every improvement that surfaces scenes the old
pool never showed.

Rows arrive in a scrambled order with `score` blanked. That is deliberate:
position would otherwise be the retriever's opinion, and a grader who reads it
grades the system instead of the scene. The order is a sort on a per-candidate
hash of `(run, query, film_slug, scene_id)` — not a shuffle of the list — so a
resumed run presents identically **and** adding or removing one candidate moves
only that candidate. That second property is what makes re-grading a changed
variant affordable as a diff rather than a full second pass. Each row carries a `pool` key recording
which variant proposed it at which rank — the template ignores it, the scorer
needs it. Without it a graded pool can only be scored as a whole.

**Only text queries actually pool.** `find` forces CLIP for image queries
regardless of `mode`, and rhyme queries call `find_rhymes` directly, so each
has exactly one producer and nothing to union with. That is not a bug — there
is no competitor to be biased against yet — but judgments on those two
modalities are slate-shaped, and will need regrading if a second image
retriever or rhyme configuration ever exists.

### Why grading is not the Annotate tab

Annotate is the closest thing in the UI by gesture, and it is the wrong
surface. Its tags feed retrieval — they are wired into the BM25 and lexical
legs. Grading through it would make your judgments an input to the system being
judged. Grading has to write somewhere retrieval never reads, which is why it
is a separate pane writing a separate file.

---

## Query file schema

Top-level `queries:` list. Per entry:

| Field | Applies to | Meaning |
|---|---|---|
| `id` | all | Stable string (`text-01`, `image-01`, `rhyme-01`) |
| `query_type` | all | `text` \| `image` \| `rhyme` |
| `text` | text | The query string |
| `image_path` | image | Repo-relative keyframe path; must exist on disk |
| `anchor` | rhyme | `<slug>/<scene_id>` |
| `lang` | all | `pt` \| `en` |
| `notes` | all | Curator-facing context |

Validated by `kuaa.eval.slates.load_modal_queries`, which raises `EvalError` on
a missing image, an unparseable anchor, or an unknown `query_type`.

**Do not add `relevant_scene_ids` / `relevance` to a query set meant for
grading.** Those are hypothesis labels — the automatic-scoring path's input,
with the failure mode described at the top of this file. They exist for
`scripts/run_eval.py`, which still requires them. Grading supersedes them.

## Authoring a query set

Write the queries **after** the corpus is indexed, against the films that are
actually there. Queries written before seeing the library reference scenes that
do not exist — and per the zero-default above, that produces a low score rather
than an error.

Cover both ends of the query-length range. `hybrid_metadata_w` tapers with
query length (`kuaa.retrieval.hybrid.effective_metadata_w`): the exact-lexical
leg earns a majority share on short object queries and actively hurts long
natural-language ones. A set of only long queries never exercises the end the
leg exists for.

Cover Portuguese properly. The BM25 corpus is English (the describer prompts
are English) and `kuaa.retrieval.bilingual` expands it at index time. PT is the
primary use case.

`corpus01_queries.yaml` is the current set and the schema reference. Its header
documents the four axes it was built on — length, language, discrimination,
surface.

---

## Automatic scoring

Kept because it produces a table with no human grades, and because the ablation
harness runs on it. Read the failure mode at the top of this file before
trusting any number it prints.

`archive_demo_queries.yaml` is the starter query set for the public demo
configured by `config/demo.yaml`, anchored to the expected demo scene order for
*The Great Train Robbery* (1903). It carries hypothesis labels because
`scripts/run_eval.py` requires them.

```bash
uv run python scripts/run_eval.py \
  --config config/demo.yaml \
  --queries data/eval/archive_demo_queries.yaml \
  --output-dir data/eval/reports
```

`scripts/run_ablation.py`'s `DEFAULT_QUERIES` points at a query set that was
deleted with the pre-corpus01 sets. Ablation needs hypothesis labels, which
`corpus01_queries.yaml` deliberately does not carry, so it currently has no
default set and must be passed `--queries` explicitly.

Reports under `data/eval/reports/` are runtime artifacts. Commit only curated
summaries meant to become release documentation.
