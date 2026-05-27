# Portfolio Launch Checklist

Updated after the rebase to `main` and the quick-wins review fixes.

This is a private working checklist, not a polished public artifact. It should
stay untracked unless we later turn it into a cleaned-up `docs/launch/` plan.

## Current State

Landed or present in the repo:

- `docs/EVALUATION_RESULTS.md` and `docs/FAILURE_ANALYSIS.md` exist.
- `docs/PERFORMANCE.md` is methodology-only now. Do not quote latency numbers
  until the production-parity benchmark is rerun on the final demo bundle.
- `scripts/run_eval.py` now defaults to configured artifact paths instead of a
  missing per-film slug.
- BM25 evaluation no longer requires CLIP embeddings for pure BM25 runs.
- CI lint/format blockers from the review are fixed locally.
- Audio search, CLIP x CLAP fusion, and reranker wiring are present in the
  codebase. The remaining portfolio work is verification, evaluation, and
  presentation, not first implementation.

Important caveat:

- The full test suite has not been rerun in this pass. We validated focused
  eval tests, ruff, black, help output, py_compile, and diff whitespace only.

## 1. Hosted Demo

Highest recruiter-impact item.

Decisions:

- Hosting target: HuggingFace Spaces is still the simplest public demo path.
- Demo bundle: use the verified public demo bundle, not local work-in-progress
  artifacts.
- Film set: default to the safest public-domain demo unless rights for larger
  material are settled.

Definition of done:

- Public URL loads a populated app.
- Text search returns ranked keyframes.
- README links to the demo in the first screen.
- Demo setup command and artifact provenance are documented.

## 2. README First Screen

Goal: a recruiter should understand the project in 30 seconds.

Keep the first screen focused on:

- what the project does;
- who it is for;
- live demo link;
- one screenshot or GIF;
- measured retrieval results;
- clear local quickstart.

Needed edit:

- Condense results from `docs/EVALUATION_RESULTS.md`.
- Add the refreshed benchmark headline only after rerunning
  `scripts/bench_retrieval.py` on the final bundle.
- Keep English first for the portfolio audience.

## 3. Demo Video And Screenshots

Still relevant. `docs/DEMO_VIDEO_SCRIPT.md` exists, but assets need to be
recorded from the current UI.

Assets to produce:

- `docs/assets/demo.mp4`
- `docs/assets/demo.gif`
- `docs/assets/screenshot-search.png`
- `docs/assets/screenshot-scene.png`
- `docs/assets/screenshot-rimas.png`

Suggested flow:

1. Open the app with the prepared demo bundle.
2. Run a text query.
3. Show hybrid/fusion behavior if the indices are present.
4. Open scene detail.
5. Show visual rhymes.
6. End with GitHub and hosted demo URL.

## 4. Public Evaluation Refresh

The repo has evaluation artifacts, but public claims should be regenerated from
the final demo bundle.

Run after the final bundle is prepared:

```bash
uv run python scripts/run_eval.py --config config/demo.yaml \
  --all-modes --top-k 10 \
  --k-rrf-sweep "10,30,60,100"
```

Then update:

- `docs/EVALUATION_RESULTS.md`
- `docs/FAILURE_ANALYSIS.md` if failure patterns changed
- README results table
- release notes draft

Do not publish numeric claims from local, stale, or partially populated
artifacts.

## 5. Performance Benchmark Refresh

The benchmark now mirrors the production hybrid path, including the current two
CLIP text encodes per hybrid query.

Run after final demo artifacts are in place:

```bash
uv run python scripts/bench_retrieval.py --n 100 --k 50 --film <film_slug>
```

Then decide whether to commit the regenerated:

- `docs/PERFORMANCE.md`
- `data/perf/bench_results.json`

Only commit benchmark output if it measures the exact bundle and hardware you
intend to describe publicly.

## 6. Audio, Fusion, And Reranker Proof

The stale "ship it or rip it" items are now replaced by proof work.

Verify:

- audio-only search works on the hosted/demo artifact set;
- CLIP x CLAP fusion works when both indices are present;
- reranker toggle is functional and does not break result cards;
- missing audio/reranker dependencies fail gracefully.

Then choose presentation scope:

- Showcase audio/fusion/reranker if they are stable in the demo.
- Otherwise label them experimental and keep the first public story focused on
  text, image, BM25, and hybrid retrieval.

Optional eval work:

- Add audio/fusion/reranker rows to the ablation only if the query labels are
  reviewed and the artifact set supports them.

## 7. Portuguese Query Gap

Still useful and not replaced by the rebase.

Measure:

- English query set vs Portuguese translation;
- same index, same top-k, same retriever;
- report R@10 and nDCG deltas.

Definition of done:

- `docs/EVALUATION_RESULTS.md` has a short "Known limitations" subsection.
- README links to the limitation instead of overclaiming multilingual quality.

## 8. AI Tooling Hygiene

Still relevant for public polish.

Review before launch:

- `docs/superpowers/`
- `.codex/`
- `.agents/`
- `CLAUDE.md`
- `CONTEXT.md`

Decision options:

- keep only if the file helps a public reviewer;
- move private planning/context files out of the public tree;
- add a short README acknowledgement that AI pair-programming was used and
  measured outcomes are documented.

Do not remove anything blindly. Some files may be useful internal records, but
they should not distract from the portfolio story.

## 9. Blog And LinkedIn Launch

Highest conversion item after the hosted demo.

Recommended first post:

> What I learned building hybrid retrieval for a Brazilian film archive: a
> measured ablation

Inputs:

- refreshed `docs/EVALUATION_RESULTS.md`;
- refreshed `docs/FAILURE_ANALYSIS.md`;
- one screenshot or GIF;
- hosted demo URL.

Definition of done:

- one blog post live;
- one LinkedIn post live;
- LinkedIn project entry points at GitHub, hosted demo, and the post.

## Suggested Order

1. Finalize and verify demo artifacts.
2. Run evaluation refresh.
3. Run performance benchmark refresh.
4. Record screenshots/video.
5. Rewrite README first screen.
6. Deploy hosted demo.
7. Clean public-facing repo hygiene.
8. Publish blog/LinkedIn launch.
