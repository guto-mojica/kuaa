# 🎞 Cinemateca imgsearch

**Sistema de catalogação audiovisual com modelos locais para acervos cinematográficos.**

Desenvolvido para cinematecas nacionais e arquivos públicos de filme, com foco em
acervos históricos de diversos períodos de produção e material de baixa resolução.

[![Licença: MIT](https://img.shields.io/badge/Licença-MIT-amber.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Versão](https://img.shields.io/badge/versão-0.3.0--dev-orange.svg)](CHANGELOG.md)

---

## Overview (English)

**Cinemateca imgsearch** is an open-source tool that turns a video file into a searchable, annotated scene catalog — entirely offline, with no data sent to external APIs.

It was built for **film archives and public cinematheques** that hold large collections of historical footage with minimal metadata. The pipeline processes a video and automatically produces:

- **Scene segmentation** — detects cuts and extracts a representative keyframe per scene
- **Visual analysis** — detects faces, objects, and classifies environment (indoor/outdoor, day/night)
- **Natural language descriptions** — a local vision model (Moondream 2) describes each scene in text
- **Semantic search** — find scenes by free-text query ("two people talking outdoors") or by a reference image, without exact keywords
- **Manual annotation** — a dedicated tab lets curators add or correct tags scene by scene, which are merged with the automated metadata for search

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

Designed for **digitised archival footage** — various production periods, variable quality, unusual aspect ratios. Runs on CPU; Apple Silicon M1+ or NVIDIA GPU recommended for the LLM step.

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

Tudo roda **localmente**, sem enviar dados para APIs externas.

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

O sistema é organizado em módulos independentes que podem ser usados separadamente:

```
src/cinemateca/
├── config.py           Configuração via YAML (default + override local)
├── device.py           Detecção de hardware (CPU / CUDA / MPS)
├── data_prep.py        Inspeção de vídeo e extração de frames (FFmpeg)
├── scene_detector.py   Detecção de cenas (PySceneDetect)
├── visual_analyzer.py  Detecção facial + objetos + ambiente (MTCNN + YOLOv8)
├── embeddings.py       Embeddings visuais e busca semântica (CLIP)
├── llm_describer.py    Descrições em linguagem natural (Moondream 2)
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

| Modelo | Tarefa | Licença | Tamanho |
|---|---|---|---|
| [CLIP ViT-B/32](https://github.com/mlfoundations/open_clip) | Embeddings visuais e busca semântica | MIT | ~150 MB |
| [Moondream 2](https://github.com/vikhyatk/moondream) | Descrição de cenas em linguagem natural | Apache 2.0 | ~1.9 GB |
| [YOLOv8n](https://github.com/ultralytics/ultralytics) | Detecção de objetos | **AGPL-3.0** | ~6 MB |
| [MTCNN](https://github.com/timesler/facenet-pytorch) | Detecção facial | MIT | ~2 MB |

> **Nota sobre o YOLOv8:** a Ultralytics usa licença AGPL-3.0.
> Para uso interno em instituições (sem redistribuição pública do software),
> isso geralmente não impõe obrigações adicionais. Consulte [SETUP.md](SETUP.md)
> para alternativas se necessário.

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
- Fonte: [Internet Archive](https://archive.org/details/paixaoflix_mazzaropi__jeca_tatu) (~~domínio público~~)
- Escolhido por representar os desafios típicos de acervo: qualidade de digitalização
  variável, variedade de ambientes (rural/urbano, interno/externo)

---

## Roadmap

### v0.1.x (correções)
- [ ] Testes de integração com vídeo de referência
- [ ] Tratamento de erros mais detalhado na interface
- [ ] Documentação de API dos módulos

### v0.2.0
- [ ] Exportação de catálogo em CSV e JSON estruturado para integração com sistemas externos
- [ ] Suporte a múltiplos vídeos no mesmo índice de busca
- [ ] Interface para comparar resultados de detecção com diferentes thresholds

### v0.3.0
- [ ] Docker image para instalação sem dependências manuais
- [ ] Classificador de ambiente treinado (substituir heurística atual)
- [ ] Suporte a legendas/intertítulos (filmes silenciosos)

### v1.0.0
- [ ] API REST para integração com sistemas de gestão de acervo existentes
- [ ] Suporte a múltiplos idiomas nos prompts do LLM
- [ ] Documentação completa em português e inglês

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
Veja [LICENSE](LICENSE) para detalhes sobre dependências.

---

## Contato e contexto institucional

Desenvolvido como ferramenta open-source para a **Cinemateca Brasileira**
e instituições parceiras de preservação audiovisual.

*Issues e Pull Requests são a forma preferida de comunicação.*
