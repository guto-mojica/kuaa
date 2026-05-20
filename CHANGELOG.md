# Changelog

Todas as mudanças notáveis neste projeto serão documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).
Versionamento segue [Semantic Versioning](https://semver.org/lang/pt-BR/):
`MAJOR.MINOR.PATCH` — sufixo `-alpha` ou `-beta` indica versões de pré-lançamento.

---

## [Não lançado]

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
`docs/superpowers/specs/2026-05-19-multimodal-retrieval-and-launch-design.md`.

#### Adicionado (implementado — branch `feat/multi-film-library`)

- **Acervo multi-filme (T1–T11)** — registry em `data/library/films.json`,
  layout per-film em `data/library/<slug>/{raw,frames,metadata,embeddings}/`,
  busca/navegação cross-film via `?film=<slug>`, seletor lateral em HTMX,
  script de migração `scripts/migrate_flat_to_library.py` (idempotente),
  pipeline com flag `--slug` que escreve sob `library/<slug>/` e registra
  o filme automaticamente. Plano detalhado em
  `docs/superpowers/plans/2026-05-20-multi-film-library.md`. Suite de
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
`docs/superpowers/specs/2026-05-17-pluggable-model-backends-design.md`.

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
