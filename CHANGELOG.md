# Changelog

Todas as mudanças notáveis neste projeto serão documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).
Versionamento segue [Semantic Versioning](https://semver.org/lang/pt-BR/):
`MAJOR.MINOR.PATCH` — sufixo `-alpha` ou `-beta` indica versões de pré-lançamento.

---

## [Não lançado]

### Removido

- **Fallback Streamlit (`app_streamlit.py`) excluído.** A interface
  FastAPI + HTMX é agora a única superfície suportada. Após o refactor
  P3 retirar `cinemateca.annotator`, o `app_streamlit.py` ficou com
  imports quebrados (`ModuleNotFoundError: cinemateca.annotator`) e
  passou a ser dead code com risco de uso acidental. Dependências
  `streamlit>=1.28` removidas do extra `full` e o extra `gui` (que
  empacotava apenas Streamlit + rich) foi excluído do `pyproject.toml`.
  Para recuperar a UI legada veja a tag `v0.2.1-streamlit-final`.

### Alterado

- **`POST /api/search/image` com upload inválido retorna 400 (antes 200).**
  Rejeições de upload (arquivo muito grande ou tipo não suportado) agora
  levantam `UserInputError` e passam pelo handler A4, retornando o envelope
  `{error, code, status}` com status 400. Antes retornavam uma página HTML
  200 com uma nota de erro inline. O upload é validado ANTES de carregar o
  índice, então o 400 é emitido mesmo sem índice no disco.

- **`GET /api/library/select/{slug}` com slug desconhecido retorna 404 (antes 200).**
  A rota agora valida o slug contra o registro `films.json`; um slug não
  registrado levanta `IndexMissing` → 404 com envelope de erro. Um slug
  válido continua retornando 200 + `HX-Redirect`. Comportamento documentado
  no docstring da rota.

### Corrigido

- **`POST /api/library/remove/{slug}` retornava 500.** A rota chamava
  `delete_film(library_dir, slug)` posicionalmente, mas a assinatura é
  `delete_film(library_dir: Path, *, slug: str)` — keyword-only após
  `*`, resultando em `TypeError: delete_film() takes 1 positional
  argument but 2 were given`. Trocado para `slug=slug`. Cobertura nova
  em `tests/test_routes_library.py` (3 testes: remove válido, remove
  com `?wipe=`, remove de slug desconhecido idempotente). Achado pela
  revisão da série P1/P2/P3.

### Adicionado

- **WS-5 — Tooling / DX / CI / Qualidade (T1–T12).** Doze tarefas de endurecimento
  do envelope de desenvolvimento sem adicionar código de produção: mypy zerou erros
  e passou a bloquear o CI (`continue-on-error` removido do job `typecheck`); gate de
  cobertura em `[tool.coverage]` com `fail_under = 75`; varredura de segurança
  `bandit` (pre-commit + CI) + `pip-audit` via `uvx` (ferramentas efêmeras, sem
  entrar em `uv.lock`); marcadores pytest (`smoke` / `acceptance` / `e2e`) + divisão
  CI smoke-rápido vs. heavy; testes de validação de config (`test_config_schema.py`)
  que exercem `Settings`, `load_config` e `ConfigError`; `justfile` de aliases `uv
  run` (`just check` = lint + type + smoke + guards); `CONTRIBUTING.md` + templates
  de PR e issue em `.github/`; `scripts/README.md` categorizando os scripts críticos
  vs. exploratórios; `scripts/verify_fresh_run.sh` — verificador reprodutível
  checkout-limpo → `uv sync` → `/health` (substituto do Docker); job de benchmark CI
  de latência de recuperação; log estruturado JSON (`cinemateca.config.loader._JsonFormatter`,
  opt-in via `json_logs: true`). Testes de regressão de configuração digitada e de
  logging em `test_config_schema.py` / `test_structured_logging.py`.

- **WS-1 — refactor profundo do núcleo de busca (C1–C11).**
  `aggregate_search` (god-function de 353 LOC) decomposto em pipeline
  componível `FilmFilter → Scorers → GlobalRRF → Materializer` (carga de
  índice única, antes dobrada), preservando comportamento byte-a-byte
  (snapshot-gated); `cfg: Any` tipado como `Settings` na superfície
  pública; caches CLIP/áudio/BM25 unificados em um `StatCache` com
  `clear_film` + contadores; `find(rerank=...)` retorna `SearchResult`
  tipado (default OFF — flip default-ON adiado para a Fase 2, dependente
  de evidência de tuning E2/E6 de WS-4); metadados por consulta no
  `SearchResult` (`fusion_used`, `reranker_applied`, `latency_ms`, …);
  `describe_batch` com checkpoint tipado (`SceneDescriptionRecord`) +
  proveniência de modelos via `ModelCard`. **Adiado por critério §10:**
  C7 (busca async — sem latência visível na escala demo de 2 filmes;
  a decomposição C1 a torna trivialmente adicionável depois), C8 (harness
  de comparação de backends — pertence a WS-4 E2, evita duplicação) e C12
  (batching — sem ganho de perfil mensurável na escala demo).

- **Tokenizador BM25 plugável (C6 / WS-1).** `cinemateca.retrieval.tokenize`
  agora exporta um `Tokenizer` Protocol (runtime-checkable), `RegexTokenizer`
  (comportamento padrão — identico ao código anterior), `MultilingualTokenizer`
  (PT-aware: remove stopwords portuguesas via nltk, opt-in) e `get_tokenizer(name)`.
  O tokenizador é injetável em `build_corpus`, `BM25Index.build`/`.query` e
  `bm25_index_for_dir`/`bm25_index_for_ctx`. Novo campo `search.bm25.tokenizer:
  regex` em `config/default.yaml` + `Bm25Cfg` no schema. O padrão `regex` mantém
  comportamento idêntico ao anterior — snapshots de regressão byte-a-byte
  confirmados sem re-record. `multilingual` ativa via `config/local.yaml`:
  `search.bm25.tokenizer: multilingual` (requer `nltk` instalado).

- **Reproducibilidade determinística (F3 / Phase 0).** Novo módulo
  `cinemateca.reproducibility` com `seed_everything(seed)` (fixa `random`,
  `numpy` global e `torch` CPU+CUDA) e `make_generator(seed, *salt)` (retorna
  um `numpy.random.Generator` salteado via CRC32, independente do PRNG
  global). Campo `seed: int = 42` adicionado à raiz de `Settings`
  (`config/schema.py`) e a `config/default.yaml`. O seed é aplicado no
  arranque do pipeline (`commands/process.py`) e no harness de avaliação
  (`scripts/run_eval.py`); é gravado no `run_manifest.json` sob `run.seed`.
  O PRNG ad-hoc `sum(ord(c))%100` em `rhymes/enrich.py:signals_for_pair`
  foi substituído por `make_generator` — as barras sintéticas do inspector
  Rimas continuam determinísticas por par (anchor, echo) mas agora respeitam
  o seed global. **Regeneração de artefactos diferida ao mantenedor** (blast
  radius quase-zero: sem `random`/`np.random` no core; MMR determinístico;
  o único PRNG era display-only e não alimenta nenhum artefacto committed).

- **Drafts de lançamento M3 (M3 #5)** sob `docs/launch/`: README
  inglês-first, outline de blog post em 7 seções (~2.300 palavras),
  scope + recipe ffmpeg para o vídeo demo de 90s. Drafts citam os
  backends realmente lançados (SigLIP2-large-256, CLAP
  larger_clap_general, bge-reranker-v2-m3, Moondream2 via HF
  transformers); placeholders de métricas (P@K, MRR, nDCG) ficam TBD
  até a tabela de ablação M4.
- **Infraestrutura de avaliação M3 pronta (M3 #3).** 50 queries curadas
  bilíngues (PT/EN) em `data/eval/m3_full_queries.yaml` cobrindo a
  distribuição spec §7.1 — 15 texto · 10 imagem · 10 áudio · 10 fusion · 5
  rimas. Subset texto-only `data/eval/m3_text_queries.yaml` (15 entradas)
  roda end-to-end via `scripts/run_eval.py` contra o índice CLIP do Jeca
  Tatu (R@5=0.189, MRR=0.254 sobre hipóteses pré-curadoria). Candidate
  slates do `/eval` gerados por `cinemateca eval seed --run m3-run-1`
  em `data/eval/m3-run-1.queries.json` (gitignored). Protocolo completo
  em `docs/EVAL_PROTOCOL.md` (~440 linhas: rubrica 0/1/2/3/S, keybindings
  do `/eval`, edge cases bilíngues, fluxo curador, política de freeze).
  Script `scripts/freeze_eval_run.sh` snapshota a JSONL de grades em
  tarball SHA256 (sanity-checks: usage error, JSONL ausente, JSONL vazio).
  Regras de `.gitignore` cobrem `data/eval/*-run-*/`,
  `*-run-*.queries.json`, `*-run-*.jsonl`, `*.frozen-*.tar.gz` — só seeds
  YAML + README ficam em git. **Sessões de anotação curatorial (Phase 3)
  pendentes** — bloqueia a ablação M4 até curadores graderem o slate. O
  harness `run_eval.py` ainda é text-only (predates multi-film); rodar as
  outras 35 queries não-texto pede um gerador de slate por modalidade
  (M3 follow-up).
- **Rimas Visuais agora diversificam via MMR rerank (M3 #2).** Novo
  popover "Diversidade" na aba Rimas com slider λ no intervalo [0, 1]
  (passo 0.05); query params `?lambda=` + `?k_candidates=` em
  `/tab/rimas`, `/api/rimas/echoes` e `/api/rimas/inspector` fazem
  override do default. Default λ=0.5 / k_candidates=30 vêm de
  `retrieval.rhymes.*` no `config/default.yaml`. Implementação:
  `mmr_rerank()` (Carbonell-Goldstein 1998) sobre embeddings CLIP
  keyframe; `find_rhymes` puxa `k_candidates` antes do rerank quando
  `λ < 1.0`. Cobertura: 6 testes unitários (`mmr_rerank`), 3 testes
  funcionais (`find_rhymes` com MMR), 4 testes de serviço, 6 testes
  de rota (3 endpoints × 2 cenários), 10 testes de aceitação em
  biblioteca real (Jeca Tatu + Edwin Porter, SigLIP2). Sliders +
  store `rimasRetrieval` localStorage-persistidos.
- **Busca fusion cross-modal (M3 #1) — `?modality=fusion&w=0.5`.**
  Combinação linear tardia de cosines CLIP × CLAP sob um peso `w`
  ajustável (slider 0..1 no popover "visual ↔ áudio"). Default 0.5;
  chip "Fusion" no Buscar. Implementação: `dispatch_fusion_search`
  espelha o dispatcher de áudio (per-film via `?film=...`, agregado
  cross-film sem `?film=`). Cobertura: 4 testes unitários em
  `search_fusion`, 8 testes de serviço (incluindo regressão da forma
  dict-of-parallel-arrays do `index_mapping.json` SigLIP2 que quebrou
  a primeira execução real), 5 testes de rota, 1 snapshot real-data
  (Jeca Tatu, skipif-guarded). `retrieval.fusion.{visual_weight, k_each}`
  no `config/default.yaml`; reranker stays opt-in (Tarefa 3.2b ainda
  pendente para plumbing live). 5 queries × 0.3/0.5/0.7 ranks pinned
  em `tests/fixtures/fusion_search_regression.json`.
- **M3 pre-flight · Busca por áudio end-to-end (CLAP)** — `/api/search`
  agora aceita `?modality=audio`, despachando para o novo módulo
  `cinemateca.search.audio` (`AudioIndex` + `load_audio_index` +
  `search_audio`) com cache por `mtime+size` por `FilmContext` (M3
  pre-flight 1.1, 1.2, 1.3). Snapshot de regressão Jeca Tatu fixado em
  `tests/fixtures/audio_search_regression.json` (1.4).
- **M3 pre-flight · UI chip de áudio** — `search.audio_enabled: true` no
  config default; chips de modalidade na superfície Buscar ligados ao
  store Alpine `buscarRetrieval`; submissão em modo áudio dispara o
  endpoint correto; strings PT/EN extraídas (Áudio/Soundtrack) e ARIA
  labels para tipo de input (M3 pre-flight 2.1, 2.2, 2.3).
- **M3 pre-flight · Cross-encoder reranker (stub preenchido)** — implementação
  real em `cinemateca.search.rerank` (default `BAAI/bge-reranker-v2-m3`;
  escape hatch `model="noop"` para hermeticidade de testes); novo wrapper
  de serviço `apply_reranker(result, *, cfg)` honra
  `cfg.retrieval.reranker.{enabled,model,top_k_in}` (M3 pre-flight 3.1,
  3.2). Chip Rerank vira affordance ao vivo na UI (3.3). Default
  `retrieval.reranker.enabled: false` — sem regressão comportamental;
  fiação nos dispatchers de produção fica como follow-up Task 3.2b.
- **M3 pre-flight · Backend visual SigLIP2-multilingual** — novo embedder
  em `cinemateca.models.clip.siglip_multilingual` (`SiglipMultilingualEmbedder`)
  registrado via `cfg.models.image_embedder` (M3 pre-flight 4.1). Acervo
  re-embeddado com `google/siglip2-large-patch16-256` (1024 dim, multilíngue,
  visão+texto) — Jeca Tatu 1236×1024 e Edwin Porter 21×1024 (4.2, 4.3).
  Backups das embeddings CLIP/OpenCLIP preservados como `.clip_openclip.npy`
  por filme para rollback rápido. SigLIP2-large-256 substituído pelo id
  inventado pelo plano (`siglip-large-multilingual` não existe na HF).
- **M3 pre-flight · CLAP archival sanity gate** — novo comando
  `cinemateca eval clap-sanity` aplica um piso P@5 sobre 5 queries
  curadas em `tests/fixtures/clap_sanity_jeca_tatu.json`; sai com código
  não-zero em FAIL para uso em pre-commit / CI; baseline atual P@5=0.60
  por query (M3 pre-flight 5.1, 5.2, 5.3).
- **M2 #3 · Busca Híbrida (CLIP ⊕ BM25 via RRF)** — `/api/search` agora aceita
  `?retriever=clip|bm25|hybrid` + pesos `sem_w` / `bm25_w`. Padrão flipado
  para `hybrid`; `?retriever=clip` reproduz a ordenação pré-M2 byte-a-byte
  (regression pin, snapshot committed em `tests/fixtures/hybrid_search_regression.json`).
  Novo pacote `src/cinemateca/retrieval/` (tokenize + corpus + bm25 + hybrid)
  alimenta o serviço `search_hybrid()` orquestrador. Aggregate cross-film
  honra o mesmo modo de retriever. Bring-forward de Alpine.js (3.14.9) para
  popovers interativos (botões Híbrido + k); Rerank/MMR ficam como chips
  read-only com micro-badges M2/M3. ~24 commits, +30 testes na suite.
  Internal design notes link the spec and plan.
- **Mojica · Frame.io redesign** — nova identidade visual (dark frio + acento
  roxo), chrome de 5 abas (adiciona **Rimas Visuais** para rimas visuais
  cross-film), abas Buscar/Cenas/Anotar/Proc redesenhadas, nova superfície
  About e novo Eval set builder atrás de flag admin. UI espelha o protótipo
  entregue em `claude_design/mojica-cinemateca/`.
- **Serviço Rimas** — kNN por cosseno cross-film sobre as embeddings CLIP de
  keyframe já existentes (MVP stub; controles de MMR/diversidade ficam para M3).
- **Eval set builder** — persistência de notas em JSONL + métricas
  P@K / nDCG / inversões / κ de Cohen; UI keyboard-first em `/eval`
  (gated por admin).
- **Camada de polish** — bus de toasts, command palette (⌘K), overlay de
  ajuda de teclado (`?`).
- **Fontes self-hosted** — Geist + JetBrains Mono offline (WOFF2).

### Mudado

- **M3 pre-flight · `Hit.rerank_score: float | None = None`** adicionado
  a `cinemateca.search.types` (campo aditivo, back-compat para callers
  existentes).
- **M3 pre-flight · `cinemateca.search.cache` + `aggregate`** agora
  despacham o embedder de imagem em query-time via
  `cinemateca.models.registry` em vez de hard-coded `OpenClipEmbedder()`;
  resolve o mismatch CLIP→SigLIP2 no caminho de busca após o re-embed
  (M3 pre-flight 4.2).
- **M3 pre-flight · `models.image_embedder` default** agora
  `siglip_multilingual` após o re-embed library-wide. Procedimento de
  rollback documentado em `docs/MIGRATIONS.md` (restaurar
  `.clip_openclip.npy` + reverter o flag de config).
- **Design tokens** — troca completa da paleta de Celluloid Amber para o
  roxo Frame.io.
- **Split de CSS** — `main.css` foi quebrado em 12 módulos (um por superfície:
  `fx`, `chrome`, `buscar`, `cenas`, `anotar`, `rimas`, `proc`, `polish`,
  `about`, `eval`, `fonts` + `main` reduzido a tokens/reset/body).
- **base.html** — novo layout de shell (TopBar de 52px + IconRail de 56px +
  LeftPane de 248px + área principal + painel direito opcional).
- **i18n** — ~280 novos msgids extraídos e traduzidos PT/EN.
- **Refactor P1 · pacote `cinemateca.search` extraído** — toda a lógica de
  busca saiu de `api/services/search.py` (1388 → **235 LOC**, teto 250) e
  `api/routes/search.py` (471 → **148 LOC**, teto 150) para um pacote novo
  em `src/cinemateca/search/` (14 arquivos: `__init__`, `types`,
  `_dispatch`, `_lookup`, `_results`, `_tag_index`, `display`, `upload`,
  `cache`, `bm25`, `clip`, `hybrid`, `aggregate`, `rerank`). API pública
  pequena e tipada: 4 verbos (`find`, `aggregate`, `reindex_bm25`,
  `rerank`) + 7 tipos (`Query`, `Filters`, `HybridWeights`, `Hit`,
  `SearchResult`, `SearchMode`, `UploadRejected`). Comportamento
  preservado byte-a-byte — verificado por 8 testes hermetic-snapshot
  (`tests/test_p1_search_snapshot.py`) sobre cenários `/api/search`
  reais (single-film CLIP/BM25/hybrid, aggregate, image upload, tag
  filter, min-sim, retriever=clip pin). ~36 novos testes unitários nos
  módulos extraídos. Spec em
  (internal design notes).
- **Refactor P2 · pacote `cinemateca.library` extraído** — fold de
  `src/cinemateca/library.py` (217 LOC) + `api/services/film_context.py`
  (138 LOC, **deletado**) + a metade data-access de `api/services/catalog.py`
  em um pacote de 6 arquivos (~620 LOC totais): `registry.py` + `scan.py` +
  `context.py` + `paths.py` + `metadata.py` + `__init__.py` (este último
  expõe o novo handle tipado `Library`). `api/services/catalog.py`:
  403 → **250 LOC** (exatamente no cap, sem mais exemption). Seis
  carve-outs `cinemateca → api.services.*` apagados de `.importlinter`
  (sobram 2, ambos P5 follow-ups: `aggregate -> services.search` e
  `_dispatch -> api.deps`). Surface pública: `from cinemateca.library
  import Library, Film, FilmContext, list_films, get_film, register_film,
  remove_film, scan_library, library_state, load_registry, save_registry,
  load_json, keyframe_url, to_smpte, derive_fps, load_tag_index,
  load_metadata` (16+ nomes em `__all__`). Comportamento preservado —
  verificado pelos 8 snapshots P1 + 17 novos testes (7 em
  `test_library_scan.py` + 10 em `test_library_handle.py`); suite total
  **774 passando**. Spec em
  (internal design notes).
- **Refactor P3 · extração de services** — três subsistemas extraídos de
  `api/services/` para pacotes em `src/cinemateca/`:

  - **`cinemateca.annotations`** (P3.A) — promove `cinemateca.annotator`
    em pacote de 4 arquivos: `io.py` + `descriptions.py` + `scenes.py` +
    `__init__.py`. Absorve a metade data-access + lógica de negócio de
    `api/services/annotations.py`. Service: 577 → **129 LOC**. Removido
    de `EXEMPTIONS`.
  - **`cinemateca.rhymes`** (P3.B) — promove `cinemateca/rhymes.py` em
    pacote de 5 arquivos: `algorithm.py` + `metadata.py` + `enrich.py` +
    `config.py` + `anchor.py`. Absorve lógica de enriquecimento +
    per-scene metadata loaders. Service: 470 → **194 LOC**. Removido de
    `EXEMPTIONS`.
  - **`cinemateca.eval` estendido** (P3.C) — adiciona `paths.py` ao
    pacote existente; estende `datasets.py` + `grades.py` +
    `grader_metrics.py` com config/data-access/IAA helpers. Service:
    564 → **244 LOC**. Removido de `EXEMPTIONS`.

  Adicionalmente: `FilmContext.from_paths` constructor adicionado +
  `Library.context` agora levanta `KeyError` (alinhado com
  `Library.get_film`; elimina workaround SimpleNamespace flagado no P2
  review) (P3.0). `.importlinter` zerado: os 2 follow-ups P5 de P1/P2
  (`aggregate -> services.search` e `_dispatch -> api.deps`) resolvidos
  movendo helpers para `cinemateca.search.aggregate` e parametrizando
  BM25 tunables como kwargs em `_dispatch.find()` (P3.D). Suite:
  **774 → 777 passando** (+3 testes). Spec:
  (internal design notes).
- **Camadas arquiteturais policiadas em CI** — `.importlinter` proíbe
  `src/cinemateca/*` de importar `api/*` (camada de núcleo HTTP-agnóstica);
  novo `scripts/check_loc_budget.py` enforça `api/services/*.py ≤ 250 LOC`
  e `api/routes/*.py ≤ 150 LOC` (com 2 exemptions documentadas para
  arquivos cujo refactor cai em P3). Ambos rodam no workflow GH Actions
  `.github/workflows/refactor-guards.yml` em todo PR.

### Documentação

- **`docs/MIGRATIONS.md`** — registro de migração de embeddings
  (OpenCLIP → SigLIP2-large-256), incluindo procedimento de rollback
  via `.clip_openclip.npy` por filme (M3 pre-flight 4.3).
- **CLAUDE.md** — tracker M2 atualizado: itens 4 (reranker), 5
  (multilingual visual model) e 6 (CLAP sanity gate) marcados como
  concluídos com notas de adaptação; item 1 (CLAP áudio na UI) marcado
  como concluído. Tracker M3 ganha bullet de pre-flight (Phases 0–5).

### Notas

- Suite de testes: **425 → 546 passando** (baseline → pós-redesign);
  **~625 → ~738 passando** após o refactor P1 (~110 testes novos no
  pacote `cinemateca.search`); **758 → 774 passando** após o refactor P2
  (+17 testes novos no pacote `cinemateca.library` — 7 em
  `test_library_scan.py` + 10 em `test_library_handle.py`).
- **32 novos commits** distribuídos em **10 fases** (Tasks 1–36) na branch
  `worktree-mojica-redesign`.
- O markup legado `.shell > .sidebar` de v0.3 segue aninhado dentro de
  `.ch-main` como wrap transicional — o unwrap final do conteúdo das abas
  (Phase 2 expansion) está adiado; o novo chrome cobre tudo visualmente e
  nenhuma regressão é exposta ao usuário final.
- **Falhas pré-existentes (NÃO regressões dos refactors P1/P2/P3):**
  `tests/test_cli_reembed_all.py::TestServe::test_serve_delegates_to_uvicorn_run`
  e `tests/test_routes_multi_film.py::TestScenesRouteMultiFilm::test_tab_scenes_unknown_slug_raises`
  falham na branch base também (verificado via `git stash` em T1 e
  re-verificado em cada T* subsequente). Tratamento fica fora do escopo
  de P1/P2/P3.
- **M3 pre-flight (Phases 0–5)** — suite final na `m3-preflight`:
  **812 passando, 2 skipped, 1 falhando** (24.2s). A única falha,
  `tests/test_search_regression_snapshot.py::test_pre_m2_snapshot_matches_or_updates`,
  é drift esperado: o snapshot fixa a ordenação pure-CLIP pré-M2 contra
  o acervo local da maintainer, e a Phase 4.3 re-embeddou tudo com
  SigLIP2 — então as embeddings subjacentes mudaram by design. Re-pin
  do snapshot (com `UPDATE_HYBRID_SNAPSHOT=1`) entra como follow-up
  trivial; não é regressão das Phases 0–5.

---

## [0.5.0-beta] — 2026-05-20

Sinais de produção, pacote de lançamento e proveniência de execução.

### Adicionado

- **Exportações estruturadas** — catálogo em JSON e CSV via pipeline e endpoints FastAPI.
- **Run manifests** — `run_manifest.json` por execução CLI e aba Processing, com proveniência completa (versão, passos, duração, artefatos).
- **Pacote de lançamento** — case study público, kit de comunicações, scripts de vídeo demo, notas de release, `scripts/check_launch_package.py` com testes de completude.
- **Demo bundle builder** — `scripts/prepare_demo.py` empacota catálogo pré-indexado para distribuição.

---

## [0.4.0-beta] — 2026-05-20

Acervo multi-filme completo (T1–T11), CLI unificada, busca aprimorada,
domain packs, framework de avaliação e demo scaffold.

### Adicionado

- **Acervo multi-filme (T1–T11)** — registry em `data/library/films.json`,
  layout por filme em `data/library/<slug>/`, busca e navegação cross-film,
  seletor de filme no sidebar (HTMX), script de migração idempotente
  (`scripts/migrate_flat_to_library.py`), pipeline com `--slug`.
- **CLI unificada** — todos os pontos de entrada sob um único app Typer
  (`cinemateca serve / process / info / library / reembed-all / config`).
- **Domain packs** — `archive` e `media_broadcast` com prompts específicos
  por domínio para os describers transformers e GGUF.
- **Framework de avaliação** — métricas de retrieval (`Recall@k`, `MRR`,
  `nDCG@10`), datasets de eval, relatório JSON/Markdown via `scripts/run_eval.py`.
- **Demo scaffold** — `config/demo.yaml`, `scripts/prepare_demo.py`,
  `data/demo/manifest.json`, docs de proveniência e verificação.
- **Busca aprimorada** — densidade 3× no índice (1:N keyframe/cena),
  deduplicação de cenas, filtro de tags agregado, observabilidade de busca.
- **Docs públicos** — ARCHITECTURE, ROADMAP, PROJECT_BRIEF, MODEL_INVENTORY,
  PRIVACY_OFFLINE, EVALUATION, DOMAIN_PACKS, TASK_BREAKDOWN.
- Estatísticas de correção de anotações.

### Corrigido

- Propagação de `?film=<slug>` do sidebar para formulário de busca e nav de abas.
- `aggregate_search` pula filmes com diretório deletado.
- Slug sanitizado na entrada do pipeline; guard para slug vazio.

---

## [0.3.0] — 2026-05-20

Conclusão da migração Streamlit → FastAPI, com paridade funcional confirmada,
estabilização da interface e backends de modelo plugáveis.

### Planejado (próximas versões)

- Exportação de catálogo em CSV e JSON estruturado
- Interface para comparar resultados de diferentes thresholds de detecção
- Detector de ambiente baseado em modelo treinado (substituir heurística atual)
- Docker image para instalação sem dependências manuais
- Testes de integração com vídeo de referência

### v1.0.0 — Lançamento público multimodal (planejado, 4 meses a partir de 2026-05-19)

Plano completo em
(internal design notes).

#### Adicionado (implementado — branch `feat/multi-film-library`)

- **Acervo multi-filme (T1–T11)** — registry em `data/library/films.json`,
  layout per-film em `data/library/<slug>/{raw,frames,metadata,embeddings}/`,
  busca/navegação cross-film via `?film=<slug>`, seletor lateral em HTMX,
  script de migração `scripts/migrate_flat_to_library.py` (idempotente),
  pipeline com flag `--slug` que escreve sob `library/<slug>/` e registra
  o filme automaticamente. Plano detalhado em
  (internal design notes). Suite de
  testes: 265 → 332 passando, 0 falhas. T12 (migração da Jeca Tatu real)
  pendente como ação manual.

#### Adicionado (planejado)

- **Acervo multi-filme** — busca e navegação atravessam todos os filmes do
  acervo (limitação honesta de filme único do v0.3.0 é removida).
- **Busca multimodal** — texto, imagem, áudio (CLAP) e transcrição (Whisper).
- **Busca híbrida** — CLIP semântico ⊕ BM25 sobre descrições/transcrições/tags
  com Reciprocal Rank Fusion.
- **Cross-encoder reranker** — re-ranking dos top-50 candidatos (padrão:
  `cross-encoder/ms-marco-MiniLM-L-12-v2`; VLM-as-judge como modo opt-in).
- **Embeddings visuais multilíngues** — consultas em PT/EN/ES, SigLIP
  multilingual (M-CLIP como fallback).
- **Fusão cross-modal** — consultas que combinam semântica visual e auditiva
  ("cena tranquila com música melancólica").
- **Rimas visuais entre filmes** — kNN sobre embeddings CLIP com restrição
  cross-film e diversidade MMR; botão "mais como esta cena, em outros filmes".
- **Framework de avaliação** — eval set privado de 50-100 pares
  (consulta, cena relevante) anotados com curadores. Métricas:
  P@5, MRR, R@20, latência por estágio, diversidade@k. Ablação completa.
- **Demonstração hospedada** — HuggingFace Spaces (CPU) com catálogo
  pré-indexado de filmes em domínio público.
- **Imagem Docker** — instalação por um comando, CPU-padrão / GPU opcional.

#### Novos Protocols (extensões a `models/base.py`)

- `AudioEmbedder` — embedding de áudio (LAION-CLAP padrão).
- `Transcriber` — transcrição com timestamps (faster-whisper padrão).

#### Artefatos públicos do lançamento

- README reescrito (herói em inglês para recrutadores; conteúdo institucional
  PT/EN preservado abaixo da dobra).
- Post técnico de ~2.500 palavras (domínio próprio canônico + artigo LinkedIn
  cross-post).
- Vídeo demo de 90 segundos (YouTube).
- Post de lançamento no LinkedIn.
- Tag `v1.0.0` no GitHub com notas de release.

#### Não-objetivos explícitos (deferidos para v2+)

OCR de legendas/intertítulos; reconhecimento facial nominal; classificador
treinado de ambiente/tipo de plano; release público do eval set;
mobile/browser-only; multi-tenant; treino de modelos foundation;
indexação distribuída.

### Ferramentas de desenvolvimento

- Adoção do **uv** para gerenciamento de ambiente e dependências.
  Dependências de desenvolvimento movidas para `[dependency-groups]`
  (PEP 735). Backend de build (`setuptools`) inalterado. Lockfile
  (`uv.lock`) adiado — instalação via `pip` continua funcionando.
- Configuração de `ruff[lint]` e correções de lint associadas; cobertura de
  testes ampliada de 18 para **208** testes (0 xfailed).

### Recuperação de regressões da interface FastAPI (v0.3.0)

Esforço de estabilização da migração Streamlit → FastAPI antes do lançamento.
Resultados visíveis para o usuário:

#### Corrigido

- **Páginas diretas** (`/search`, `/scenes`, `/annotate`, `/processing`,
  `/about`) voltam a renderizar a página completa, em paridade com a troca de
  abas via HTMX — não mais um fragmento solto.
- **Aba Processamento** voltou a renderizar (correção do filtro `split` no
  template e do checklist de etapas).
- **Filtro de tags em Cenas** passa a casar corretamente quando o id da cena é
  inteiro, string ou misto (normalização consistente).
- **Streaming de Processamento (SSE)** encerra de forma limpa: sequência tipada
  `update` → `done`, frame único de `error`/`cancelled`, sem reconexão infinita.

#### Adicionado / Melhorado

- Camada de serviços extraída para `api/services/` (catálogo, anotações, busca)
  com salvamento atômico de anotações, validação/cache por mtime do índice e
  checagens de upload de imagem.
- Cancelamento de job no pipeline, integrado ao bloqueio por dependências
  descrito em **Alterado** (etapa anterior que falha bloqueia as seguintes com
  motivo explícito; sem saída parcial apresentada como sucesso).
- **Acervo de um único filme** honesto na barra lateral (estado global real;
  suporte a múltiplos filmes permanece adiado, ver lista acima).
- **i18n / acessibilidade / offline**: rótulos de etapas traduzíveis, idioma e
  caminho cientes da localidade, `href`s reais para navegação sem JS, ícones
  Phosphor e fontes IBM Plex vendorizados (sem CDN, funciona totalmente offline).

Checklist de verificação manual pré-lançamento em
`docs/RELEASE_VERIFICATION.md`.

### Alterado

- A execução do pipeline pela aba **Processamento** agora usa
  **bloqueio por dependências**: uma etapa cujos pré-requisitos falharam,
  foram bloqueados ou cujos artefatos de entrada estão ausentes é marcada
  como `blocked` (com o motivo) e não é executada, em vez de consultar
  `pipeline.stop_on_error`. Isso elimina o defeito histórico em que uma
  falha em `scene_detection` ainda permitia `embeddings`/`llm_description`
  rodarem sobre um `keyframes_metadata.json` obsoleto. O comportamento
  padrão não muda (`stop_on_error: false`); a chave
  `pipeline.stop_on_error` continua valendo para o CLI
  `python -m cinemateca process`, que mantém a parada no primeiro erro
  quando habilitada.

### Backends de modelo plugáveis (PROTOCOL_OPTION Passos 1–2)

Refatoração que torna cada papel de modelo do pipeline substituível por
configuração. Ver `docs/PROTOCOL_OPTION.md` e
(internal design notes).

#### Adicionado / Melhorado

- **5 `Protocol`s tipados** (`ImageEmbedder`, `FaceDetector`,
  `ObjectDetector`, `SceneDescriber`, `EnvironmentClassifier`) com um
  **registry orientado por configuração** em `src/cinemateca/models/`.
  O pipeline importa apenas dos Protocols / do registry, nunca de um
  backend concreto.
- **`by_image`** desacoplado dos internos do embedder via
  `encode_image_single`.
- **Knob de offload de GPU orientado por configuração** (`llm.gpu_layers`,
  padrão `-1`) e log por cena (resposta + tags) no describer GGUF.

#### Corrigido

- **Descritor de cenas keyless e offline**: `transformers` e o Moondream
  sobre PyTorch foram removidos; o describer agora usa **Moondream 2 GGUF**
  via `llama-cpp-python` (sem chave de API, sem nuvem). `pyvips`/`einops`
  deixam de ser necessários — isso corrige a falha original
  `"No module named 'pyvips'"`.
- **Aliases de `--steps`**: as formas curtas
  `frames`/`scenes`/`visual`/`embeddings`/`llm` voltam a resolver para os
  nomes canônicos das etapas.

#### Notas técnicas

- Artefatos LLM de **Jeca Tatu (1959)** regenerados integralmente pelo novo
  caminho do registry GGUF: **412 cenas, 0 linhas de erro, 0 descrições
  vazias, 737 tags** (gate de aceitação aprovado). Isso também validou a
  correção do bug de resume sobre o arquivo real corrompido (275 erros).
- **Limitação conhecida — aceleração por GPU não está ativa.** Compilar um
  `llama-cpp-python` com CUDA nesta máquina está bloqueado por uma
  incompatibilidade glibc 2.43 (Fedora 44) ↔ CUDA 13.0.2 (conflito de
  exception-spec em `rsqrt`/`rsqrtf` no `crt/math_functions.h` da CUDA,
  corrigido pela NVIDIA apenas na CUDA 13.1, indisponível aqui). O knob
  `llm.gpu_layers` já está implementado e ativará a GPU sem nenhuma
  mudança de código assim que um build de `llama-cpp-python` com
  CUDA ≥ 13.1 for instalado. A regeneração rodou em CPU.

---

## [0.1.0-alpha] — 2025

### Primeira versão pública de protótipo

Esta versão marca a transição dos notebooks Jupyter de prototipagem
para uma aplicação estruturada com módulos reutilizáveis e interface gráfica.

#### Adicionado

**Módulos Python (`src/cinemateca/`)**
- `config.py` — carregamento de configuração via YAML com merge de defaults
- `device.py` — detecção automática de hardware (CPU / CUDA / Apple Silicon MPS)
- `data_prep.py` — inspeção de vídeo via FFprobe e extração de frames via FFmpeg
- `scene_detector.py` — detecção de cenas com PySceneDetect (ContentDetector e AdaptiveDetector), extração de keyframes e exportação de metadados em JSON
- `visual_analyzer.py` — detecção facial (MTCNN), detecção de objetos (YOLOv8n) e classificação de ambiente (heurística de brilho/bordas)
- `embeddings.py` — geração de embeddings visuais com CLIP ViT-B/32 e busca semântica por texto, imagem ou combinada (texto + filtro de tags)
- `llm_describer.py` — geração automática de descrições, tags e metadados estruturados com Moondream 2; processamento com checkpoints para retomada
- `pipeline.py` — orquestrador completo com skip de etapas existentes, controle de erros e relatório de execução

**Interface gráfica (`app.py`)**
- Tab **Processar**: upload de vídeo ou caminho local, seleção de etapas, controles de detecção de cenas (algoritmo e threshold), execução com status em tempo real
- Tab **Pesquisar**: busca semântica por texto e por imagem de referência, filtro por tags LLM, grid de resultados com metadados
- Tab **Catálogo**: navegação paginada de todas as cenas, filtros por localização, período do dia e número de pessoas

**CLI (`python -m cinemateca`)**
- `process --video <path>` — pipeline completo ou etapas selecionadas
- `info --video <path>` — propriedades técnicas do vídeo

**Configuração**
- `config/default.yaml` — todos os parâmetros documentados com valores padrão
- Sistema de override via `config/local.yaml` (não versionado)

**Projeto**
- `pyproject.toml` com grupos de dependências opcionais (`[full]`, `[search]`, `[gui]`, `[dev]`)
- `tests/test_smoke.py` com 18 testes cobrindo imports, config e parsing
- `SETUP.md` com guia de instalação para macOS e Linux
- `LICENSE` MIT com nota sobre dependência AGPL (YOLOv8)

#### Corrigido (em relação aos notebooks originais)

- Inconsistência de caminhos entre notebooks (NB01 usava `cwd().parent`, NB04/05 usavam `cwd()`) — unificado via `config.project_root`
- Células do NB02 armazenadas como listas de caracteres individuais — código reassemblado corretamente
- CLIP e Moondream carregados em duplicata na busca combinada do NB05 — agora um único `CLIPEmbedder` é reutilizado no pipeline
- `FORCE_CPU = True` hardcoded no NB04/05 — movido para `config.hardware.force_cpu`
- Nome de arquivo de metadados sem contexto do filme (`video_properties.json`) — resolvido via config de caminhos

#### Notas técnicas

- Filme de teste: **Jeca Tatu (1959)**, Mazzaropi — P&B, ~90min, domínio público
- Hardware de desenvolvimento: Apple Silicon (MPS disponível mas CPU usada por estabilidade)
- Moondream 2 revisão `2025-01-09` — pinada explicitamente para reprodutibilidade
- Busca semântica sem FAISS (produto escalar NumPy) — elimina dependência complexa sem perda significativa de performance para acervos de até ~10.000 cenas

---

*Para reportar bugs ou sugerir funcionalidades, abra uma Issue no repositório.*
