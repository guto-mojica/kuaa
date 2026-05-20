# Cinemateca imgsearch

**Offline multimodal search and metadata generation for archival video collections.**

Cinemateca imgsearch is a local-first applied AI workbench that turns video files
into searchable, human-reviewable scene catalogs. The first domain is film
archive cataloging: historical footage, sparse metadata, unusual aspect ratios,
and digitized material with variable quality.

Portuguese context: sistema de catalogação audiovisual com modelos locais para
cinematecas nacionais e arquivos públicos de filme.

[![Licença: MIT](https://img.shields.io/badge/Licença-MIT-amber.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Versão](https://img.shields.io/badge/versão-0.3.0--dev-orange.svg)](CHANGELOG.md)

---

## Overview (English)

**Cinemateca imgsearch** is an open-source tool for private visual collections.
After installation and model downloads, processing and search run locally, with
no video, keyframes, annotations, embeddings, or search queries sent to hosted AI
APIs.

It was built for **film archives and public cinematheques** that hold large
collections of historical footage with minimal metadata. The pipeline processes
a video and produces:

- **Scene segmentation** — detects cuts and extracts a representative keyframe per scene
- **Visual analysis** — detects faces, objects, and classifies environment (indoor/outdoor, day/night)
- **Natural language descriptions** — a local vision model (Moondream 2) describes each scene in text
- **Semantic search** — find scenes by free-text query ("two people talking outdoors") or by a reference image, without exact keywords
- **Manual annotation** — a dedicated tab lets curators add or correct tags scene by scene, which are merged with the automated metadata for search

### Who this is for

- **Archivists and curators** who need a searchable first pass over digitized film collections.
- **Researchers** who need to find visual moments inside long-form footage.
- **Applied AI reviewers** who want to inspect a realistic local multimodal system rather than a hosted API demo.

Future domain packs are planned for adjacent private visual-search workflows,
such as media asset review and inspection footage. The current implemented
product remains archive-first.

### Current status

Implemented now:

- Single-machine video processing pipeline.
- FastAPI + HTMX web interface.
- Text search, image search, scene browsing, and manual annotation.
- Configurable local model backends using typed Protocols.
- Offline-oriented static assets and local artifact storage.
- Regression tests for the web/service/pipeline surfaces.

Planned next:

- Reproducible public demo data and precomputed artifacts.
- Retrieval and metadata-quality evaluation.
- Domain pack configuration, starting with the current archive domain.
- Structured exports, run manifests, API docs, and stronger packaging.

### Project docs

These documents explain the public positioning, architecture, model stack, and
portfolio roadmap:

| Document | Purpose |
|---|---|
| [Project brief](docs/PROJECT_BRIEF.md) | Problem framing, users, positioning, and portfolio value |
| [Architecture](docs/ARCHITECTURE.md) | Pipeline, web app, model registry, artifacts, and constraints |
| [Model inventory](docs/MODEL_INVENTORY.md) | Model roles, backends, licenses, download behavior, and risks |
| [Offline and privacy notes](docs/PRIVACY_OFFLINE.md) | What stays local, when network access may happen, and safe public claims |
| [Portfolio implementation plan](docs/PORTFOLIO_IMPLEMENTATION_PLAN.md) | Phased plan for demo, evaluation, domain packs, and launch |
| [Task breakdown](docs/TASK_BREAKDOWN.md) | Issue-sized tasks derived from the implementation plan |
| [Roadmap](docs/ROADMAP.md) | Short public roadmap snapshot |

### Quick start

```bash
git clone https://github.com/guto-mojica/cinemateca-imgsearch.git
cd cinemateca-imgsearch
uv venv
uv sync --extra full --group dev
uv run app.py                 # FastAPI + HTMX, opens at http://localhost:8501
# Legacy Streamlit UI (during migration): uv run streamlit run app_streamlit.py
```

The web interface (FastAPI + HTMX) has these tabs:

| Tab | Purpose |
|---|---|
| **Buscar** | Semantic search by text or reference image |
| **Cenas** | Browse and filter all catalogued scenes |
| **Anotar** | Manually add or correct tags on individual scenes |
| **Processamento** | Pipeline progress — appears only while a video is processing |

Designed for **digitized archival footage**: various production periods,
variable quality, unusual aspect ratios. Runs on CPU; Apple Silicon M1+ or an
NVIDIA GPU is recommended for the vision-language model step.

---

## O que é

Cinemateca imgsearch é uma ferramenta open-source que processa um arquivo de vídeo e gera
automaticamente um catálogo pesquisável com:

- **Segmentação de cenas** — detecta cortes e extrai um keyframe representativo de cada cena
- **Análise visual** — identifica rostos, objetos e classifica ambiente (interno/externo, dia/noite)
- **Descrições em linguagem natural** — um modelo de visão local descreve cada cena em texto
- **Busca semântica** — encontra cenas por texto ("dois personagens conversando do lado de fora")
  ou por imagem de referência, sem palavras-chave exatas
- **Metadados estruturados** — timecodes, tags, contagem de pessoas, objetos — prontos para
  integração com sistemas de gestão de acervo

Após instalação e download dos modelos, o processamento e a busca rodam
**localmente**. Vídeos, keyframes, anotações, embeddings e consultas não precisam
ser enviados para APIs externas.

---

## Por que este projeto existe

Cinematecas e arquivos públicos ao redor do mundo têm acervos de milhares de filmes
que só existem como descrições manuais — às vezes apenas o título, o ano, duração e ficha técnica.
A catalogação detalhada de uma coleção grande é inviável manualmente.

Este sistema não substitui o trabalho curatorial humano. Ele gera um primeiro nível
de metadados que:

1. Torna o acervo pesquisável *antes* da catalogação manual
2. Acelera o trabalho dos catalogadores ao apresentar contexto visual imediato
3. Funciona bem com material de baixa qualidade — o foco de design são filmes digitalizados
   de arquivo, não produções contemporâneas em alta resolução

---

## Pré-requisitos

- Python 3.10+
- FFmpeg instalado no sistema
- 16 GB RAM (mínimo); 32 GB recomendado para o módulo LLM
- ~20 GB de espaço em disco (modelos + dados de um filme de teste)
- Hardware: CPU moderna (suficiente), Apple Silicon M1+ ou GPU NVIDIA (recomendado para LLM)

---

## Instalação rápida

```bash
# 1. Clonar o repositório
git clone https://github.com/guto-mojica/cinemateca-imgsearch.git
cd cinemateca-imgsearch

# 2. Criar o ambiente (uv usa a versão fixada em .python-version)
uv venv

# 3. Instalar dependências (extra "full" + grupo de dev)
uv sync --extra full --group dev
# Sem uv: python3 -m venv .venv && source .venv/bin/activate
#         && pip install -e ".[full]" && pip install pytest pytest-cov black ruff mypy

# 4. Iniciar a interface (FastAPI + HTMX)
uv run app.py
# Interface Streamlit legada (durante a migração):
#   uv run streamlit run app_streamlit.py
```

Para instruções detalhadas, incluindo instalação do FFmpeg e configuração para
servidores remotos, veja [SETUP.md](SETUP.md).

---

## Uso

### Interface web

```bash
uv run app.py
# Abre em http://localhost:8501 (FastAPI + HTMX)
```

A interface tem as abas:

| Aba | Função |
|---|---|
| **Buscar** | Busca semântica por texto ou imagem no acervo indexado |
| **Cenas** | Navegar e filtrar todas as cenas catalogadas |
| **Anotar** | Adicionar ou corrigir tags manualmente, cena a cena |
| **Processamento** | Progresso do pipeline — aparece apenas durante o processamento |

### Linha de comando

```bash
# Inspecionar um vídeo
uv run python -m cinemateca info --video caminho/para/filme.mp4

# Processar um vídeo completo
uv run python -m cinemateca process --video caminho/para/filme.mp4

# Processar com configuração personalizada
uv run python -m cinemateca process --video caminho/para/filme.mp4 --config config/local.yaml

# Executar apenas etapas específicas
uv run python -m cinemateca process --video caminho/para/filme.mp4 --steps scenes,embeddings
```

### Como módulo Python

```python
from cinemateca.config import load_config, setup_logging
from cinemateca.pipeline import CatalogPipeline

cfg = load_config("config/local.yaml")
setup_logging(cfg)

pipeline = CatalogPipeline(cfg)
result = pipeline.run("data/raw/meu_filme.mp4")
print(result.summary())
```

---

## Arquitetura

O sistema é organizado em módulos independentes que podem ser usados separadamente.
Os backends de modelo ficam atrás de `Protocol`s tipados e são selecionados por
configuração em `src/cinemateca/models/registry.py`.

```
src/cinemateca/
├── config.py           Configuração via YAML (default + override local)
├── device.py           Detecção de hardware (CPU / CUDA / MPS)
├── data_prep.py        Inspeção de vídeo e extração de frames (FFmpeg)
├── scene_detector.py   Detecção de cenas (PySceneDetect)
├── visual_analyzer.py  Fachada de análise visual com backends injetados
├── embeddings.py       Busca semântica NumPy sobre embeddings CLIP
├── models/             Protocols + registry + backends concretos
│   ├── clip/            OpenCLIP
│   ├── describer/       Moondream 2 via transformers ou GGUF
│   ├── environment/     Heurística OpenCV
│   ├── face/            MTCNN
│   └── objects/         YOLOv8
└── pipeline.py         Orquestrador do pipeline completo
```

**Pipeline de dados:**

```
Vídeo
  │
  ├─[FFmpeg]──────────► frames/         (1 frame/segundo)
  │
  ├─[PySceneDetect]──► keyframes/       (1 keyframe por cena)
  │                     metadata/keyframes_metadata.json
  │
  ├─[MTCNN + YOLO]───► metadata/visual_analysis.json
  │
  ├─[CLIP]────────────► embeddings/     (vetores para busca semântica)
  │
  └─[Moondream 2]────► metadata/scene_descriptions.json
                        metadata/scene_tags.json
```

---

## Modelos utilizados

| Modelo | Tarefa | Backend atual | Observação |
|---|---|---|---|
| [CLIP ViT-B/32](https://github.com/mlfoundations/open_clip) | Embeddings visuais e busca semântica | OpenCLIP | Pesos podem baixar no primeiro uso |
| [Moondream 2](https://huggingface.co/vikhyatk/moondream2) | Descrição de cenas em linguagem natural | transformers por padrão; GGUF opcional | Revisão fixada em config para reprodutibilidade |
| [YOLOv8n](https://github.com/ultralytics/ultralytics) | Detecção de objetos | Ultralytics | Licença AGPL/Enterprise exige atenção |
| [MTCNN](https://github.com/timesler/facenet-pytorch) | Detecção facial/contagem | facenet-pytorch | Detecção, não reconhecimento de identidade |

> **Nota sobre o YOLOv8:** a Ultralytics usa licença AGPL-3.0.
> Para publicação, redistribuição, uso institucional ou uso comercial, verifique
> as obrigações da licença, obtenha a licença adequada ou substitua/desative esse
> backend. Veja também [Model inventory](docs/MODEL_INVENTORY.md).

---

## Configuração

Todos os parâmetros são controlados via `config/default.yaml`.
Para personalizar sem modificar os defaults:

```bash
cp config/default.yaml config/local.yaml
# Edite config/local.yaml com seus caminhos e preferências
```

`config/local.yaml` nunca é versionado — cada instalação tem o seu.

Os parâmetros mais relevantes para ajuste inicial:

```yaml
hardware:
  device: "auto"         # "cpu", "cuda", "mps", ou "auto"

frame_extraction:
  fps: 1                 # frames por segundo a extrair
  sample_duration: null  # null = vídeo inteiro; ex: 300 = primeiros 5min

scene_detection:
  detector: "content"        # "content" ou "adaptive"
  content_threshold: 27.0    # menor = mais cenas detectadas
```

---

## Filme de teste

O desenvolvimento usa **Jeca Tatu (1959)** de Amácio Mazzaropi como referência:

- Formato: P&B, ~90 minutos
- Fonte de desenvolvimento: [Internet Archive](https://archive.org/details/paixaoflix_mazzaropi__jeca_tatu)
- Status de direitos: verificar antes de redistribuir vídeo, keyframes ou artefatos derivados
- Escolhido por representar os desafios típicos de acervo: qualidade de digitalização
  variável, variedade de ambientes (rural/urbano, interno/externo)

---

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the current public roadmap and
[docs/PORTFOLIO_IMPLEMENTATION_PLAN.md](docs/PORTFOLIO_IMPLEMENTATION_PLAN.md)
for the portfolio-oriented implementation plan.

Near-term focus:

- reproducible public demo data,
- evaluation metrics for retrieval and metadata quality,
- domain pack configuration,
- structured exports and run manifests.

---

## Contribuindo

Contribuições são bem-vindas. Para mudanças significativas, abra uma Issue
primeiro para discutir o que você gostaria de modificar.

Para rodar os testes:
```bash
uv sync --group dev
uv run pytest tests/
```

---

## Licença

MIT — veja [LICENSE](LICENSE) para o texto completo.

Nota: o módulo de detecção de objetos usa YOLOv8 (AGPL-3.0).
Veja [LICENSE](LICENSE) e [Model inventory](docs/MODEL_INVENTORY.md) para
detalhes sobre dependências.

---

## Contato e contexto institucional

Desenvolvido como ferramenta open-source para a **Cinemateca Brasileira**
e instituições parceiras de preservação audiovisual.

*Issues e Pull Requests são a forma preferida de comunicação.*
