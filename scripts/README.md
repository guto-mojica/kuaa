# scripts/

Operational and analysis scripts. Two tiers.

## Critical (CI / release / operations — do not delete without updating callers)

| Script | Role | Invoked by |
|---|---|---|
| `check_loc_budget.py` | LOC budget gate (services ≤250, routes ≤150) | `.github/workflows/refactor-guards.yml` |
| `check_launch_package.py` | Public-docs gate: headings, promised links, zero placeholder tokens | `.github/workflows/ci.yml` (`docs` job), `just docs` |
| `bench_retrieval.py` | Latency p50/p95/p99 per retriever | `.github/workflows/ci.yml` (`benchmark` job, non-blocking), `just bench` |
| `run_eval.py` | Retrieval eval (clip/bm25/hybrid; `--all-modes`) | `just eval`; `SETUP.md` §7, `docs/EVALUATION.md` |
| `run_ablation.py` | Retriever-variant ablation; regenerates `docs/EVALUATION_RESULTS.md` | run by hand; see `docs/EVALUATION.md` |
| `verify_reranker.py` | E2E cross-encoder reranker on real artefacts | release verification |
| `verify_fresh_run.sh` | Clean checkout → `uv sync` → boot → `/health` | `just verify`; run before a release |
| `build_demo_bundle.py` | Deterministic demo ZIP | demo release |
| `prepare_demo.py` | Download/validate demo bundle | `SETUP.md` §7; CI `benchmark` job (best-effort) |
| `freeze_eval_run.sh` | SHA256-tar eval grades for provenance | run by hand before citing a graded run |
| `migrate_flat_to_library.py` | One-shot v0.3 flat→per-film migration | historical, run-once |
| `ensure_gpu_llama.sh` | Rebuild CUDA llama-cpp after `uv sync` | GPU describer ops; see `docs/GPU_LLAMA_CPP_CUDA_BUILD.md` |

## Exploratory

One-off analysis scripts are deliberately kept out of the tracked tree.
`.gitignore` reserves their paths — `analyze_failures.py`, `analyze_pt_gap.py`,
`analyze_pt_gap_evaluable.py` — so a local copy is never committed by accident.
Nothing in CI or the release path depends on them.

Run any script with `uv run python scripts/<name>.py --help` (Python) or
`bash scripts/<name>.sh` (bash). All paths resolve relative to the repo root.
