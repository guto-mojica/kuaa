# SETUP.md — KUAA

Guia de instalação e execução para sistemas Unix (macOS e Linux).  
Tempo estimado: 15–30 minutos (dependendo da velocidade de download dos modelos).

---

## Índice

1. [Pré-requisitos](#1-pré-requisitos)
2. [Obter o código](#2-obter-o-código)
3. [Ambiente Python](#3-ambiente-python)
4. [Instalar dependências](#4-instalar-dependências)
5. [Instalar FFmpeg](#5-instalar-ffmpeg)
6. [Configurar caminhos](#6-configurar-caminhos)
7. [Executar a interface](#7-executar-a-interface)
8. [Executar pelo terminal (CLI)](#8-executar-pelo-terminal-cli)
9. [Primeira execução — o que esperar](#9-primeira-execução--o-que-esperar)
10. [Solução de problemas comuns](#10-solução-de-problemas-comuns)
11. [Atualizar para uma nova versão](#11-atualizar-para-uma-nova-versão)

---

## 1. Pré-requisitos

Antes de começar, verifique que seu sistema tem:

| Requisito | Mínimo | Recomendado |
|---|---|---|
| Python | 3.10 | 3.11 ou 3.12 |
| RAM | 16 GB | 32 GB |
| Armazenamento livre | 20 GB | 50 GB |
| Hardware | CPU moderna | Apple Silicon M1+ ou GPU NVIDIA |

**Verificar versão do Python:**
```bash
python3 --version
# Deve retornar Python 3.10.x ou superior
```

Se não tiver Python 3.10+, instale via [python.org](https://www.python.org/downloads/)
ou, em macOS, via Homebrew:
```bash
brew install python@3.11
```

---

## 2. Obter o código

**Opção A — clonar do GitHub** (quando o repositório estiver publicado):
```bash
git clone https://github.com/guto-mojica/kuaa.git
cd kuaa
```

**Opção B — a partir de um arquivo .zip** (distribuição direta):
```bash
unzip kuaa.zip
cd kuaa
```

A estrutura que você deve ver:
```
kuaa/
├── app.py                  ← entrypoint FastAPI (uvicorn api.server:app)
├── pyproject.toml          ← definição do pacote e dependências
├── uv.lock                 ← versões fixadas (uv)
├── justfile                ← atalhos de desenvolvimento (opcional)
├── api/                    ← rotas HTTP (FastAPI)
├── src/kuaa/               ← núcleo de IA (search, library, rhymes, eval, ...)
├── web/                    ← templates Jinja2, HTMX e i18n
├── config/                 ← default.yaml (versionado) e local.yaml (seu override)
├── docs/                   ← documentação de design, arquitetura e operação
├── scripts/                ← demo, avaliação e verificação
├── site/                   ← site público (GitHub Pages)
└── tests/                  ← testes (pytest)
```

---

## 3. Ambiente Python

É fortemente recomendado usar um **ambiente virtual** — um Python isolado
só para este projeto, que não interfere com o Python do sistema.

```bash
# uv cria e gerencia o ambiente automaticamente (faça isso UMA vez):
uv venv
# uv usa a versão fixada em .python-version (3.11). Se não tiver o
# Python 3.11, o uv baixa automaticamente.

# Não é necessário "ativar" o ambiente: use `uv run <comando>`.
# Se preferir ativar manualmente mesmo assim:
# macOS / Linux:
source .venv/bin/activate
```

**Com uv não é necessário ativar o ambiente:** basta prefixar os
comandos com `uv run` (ex.: `uv run python ...`). Se preferir ativar
manualmente, lembre de reativar a cada novo terminal:
```bash
cd kuaa
source .venv/bin/activate   # opcional — só se não usar `uv run`
```

Para desativar um ambiente ativado manualmente:
```bash
deactivate
```

---

## 4. Instalar dependências

Com o ambiente criado (`uv venv`):

```bash
# Instalar o pacote + todas as dependências (extra "full") + ferramentas
# de desenvolvimento (grupo "dev"):
uv sync --extra full --group dev

# Sem uv (alternativa, não suportada): o pip ignora o uv.lock, então
# as versões instaladas podem diferir das testadas:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -e ".[full]" && pip install pytest pytest-cov ruff pre-commit mypy httpx import-linter
```

O que `-e ".[full]"` significa:
- `-e` = "editable install" — o código em `src/` é usado diretamente,
  sem precisar reinstalar a cada alteração
- `[full]` = instalar todas as dependências opcionais (torch, CLIP,
  YOLOv8, FastAPI, etc.)

Este comando vai baixar aproximadamente **2–4 GB** de pacotes na
primeira vez. As execuções seguintes são instantâneas.

> **Não rode comandos `uv` que mutam o ambiente em paralelo.** Um
> `uv sync`/`uv venv` concorrente com outro `uv run` pode deixar o
> `.venv` num estado parcial (ex.: avisos de `RECORD` ausente). Se isso
> acontecer, rode `uv sync --extra full --group dev` novamente de forma
> isolada para reparar o ambiente.

**Se você não precisa de todos os módulos** (por exemplo, só quer a
busca semântica sem o módulo LLM):
```bash
uv sync --extra search    # só CLIP e busca
uv sync --extra web       # só a interface FastAPI + HTMX
```

**Ferramentas opcionais:**

- Hooks de pre-commit (ruff, etc.), instalados no seu clone:
  ```bash
  uv run pre-commit install
  ```
- Testes de interface no navegador (grupo `e2e`, Playwright). Não entram
  na instalação padrão:
  ```bash
  uv sync --group e2e
  uv run playwright install chromium
  ```
- Atalhos do projeto estão no arquivo `justfile`. Para usá-los, instale o
  programa `just` e leia os nomes das receitas no arquivo.

**Verificar instalação:**
```bash
uv run python -c "import kuaa; print(kuaa.__version__)"
# Deve imprimir: 0.11.0
```

---

## 5. Instalar FFmpeg

FFmpeg é usado para inspeção de vídeo e é a base de leitura de frames do PySceneDetect na detecção de cenas.
Ele não é um pacote Python — precisa ser instalado no sistema.

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Fedora / RHEL:**
```bash
sudo dnf install ffmpeg
```

**Verificar:**
```bash
ffmpeg -version
ffprobe -version
# Ambos devem retornar informações de versão, não "comando não encontrado"
```

---

## 6. Configurar caminhos

O sistema usa `config/default.yaml` como base. Para ajustar caminhos
ou parâmetros, crie um arquivo de override **sem modificar o default**:

```bash
cp config/default.yaml config/local.yaml
```

Edite `config/local.yaml` com suas preferências. Exemplo mínimo:

```yaml
# config/local.yaml — só inclua o que você quer sobrescrever

paths:
  data_dir: "/dados/acervo/kuaa"   # onde ficam os vídeos e frames
  outputs_dir: "/dados/acervo/outputs"

hardware:
  device: "auto"     # "auto", "cpu", "cuda", ou "mps"
```

`config/local.yaml` **nunca deve ser commitado no git** — ele fica
local em cada máquina. O `.gitignore` já está configurado para isso.

---

## 7. Executar a interface

Com o ambiente virtual ativado e dependências instaladas:

```bash
# Demo público com artefatos pré-computados
uv run python scripts/prepare_demo.py --download
KUAA_CONFIG=config/demo.yaml uv run kuaa serve

# Avaliação do índice demo preparado
uv run python scripts/run_eval.py --config config/demo.yaml \
  --queries data/eval/archive_demo_queries.yaml \
  --output-dir data/eval/reports

# Interface principal FastAPI + HTMX
uv run kuaa serve
```

`kuaa serve` aceita só `--host`, `--port`, `--reload` e `--no-reload`.
O arquivo `config/local.yaml`, se existir, é aplicado automaticamente
(veja `uv run kuaa config show`). Para usar outra configuração na
interface, defina a variável `KUAA_CONFIG` antes do comando.

Acesse manualmente: **http://localhost:8501**

Para detalhes do demo público, veja `docs/DEMO.md`. Para métricas de busca,
veja `docs/EVALUATION.md`.

### Domain packs

O domínio padrão é `archive`, definido em `config/domains/archive.yaml`.
Para experimentar outro domínio, aponte a configuração para um pack diferente:

```yaml
domain:
  pack: "media_broadcast"
  packs_dir: "config/domains"
```

Os packs mudam prompts, campos e mapeamento de exportação sem alterar o
pipeline. Veja `docs/DOMAIN_PACKS.md`.

### Verificar uma instalação limpa (sem Docker)

Para provar que um clone novo sobe com apenas `uv`:

```bash
uv sync --extra full --group dev
uv run kuaa serve            # http://127.0.0.1:8501
# Em outro terminal:
curl -fsS http://127.0.0.1:8501/health
```

Ou rode o verificador automatizado (clone efêmero → sync → boot → /health):

```bash
scripts/verify_fresh_run.sh
```

O script arquiva o HEAD atual, cria um `.venv` isolado no diretório temporário,
sobe o servidor e aguarda `GET /health` retornar 200. Flags úteis:
- `--keep` — mantém o diretório temporário para inspeção
- `--port N` — porta do servidor (padrão: 8599)
- `--timeout SECONDS` — tempo máximo de espera (padrão: 180)
- `--extra EXTRA` — extras `uv` a instalar (padrão: `web`, que basta para
  `/health`; use `full` para verificar a cadeia de importação completa)

### Exports and run manifests

Depois que um catálogo existir, a interface pode exportar JSON e CSV:

```bash
curl -L http://localhost:8501/api/export/catalog.json -o catalog_export.json
curl -L http://localhost:8501/api/export/catalog.csv -o catalog_export.csv
```

Cada execução do pipeline grava `run_manifest.json` junto dos metadados,
registrando entrada, hash de config, domínio, modelos, etapas e artefatos.
Veja `docs/API.md` e `docs/OPERATIONS.md`.

**Para rodar em uma máquina remota** (servidor da kuaa) e
acessar do seu computador:
```bash
# Na máquina remota:
uv run kuaa serve --host 0.0.0.0 --port 8501

# No seu navegador local:
# http://IP-DA-MAQUINA-REMOTA:8501
```

**Para manter rodando após fechar o terminal** (em servidores):
```bash
# Usando nohup (simples). A pasta logs/ não é versionada, então crie-a antes:
mkdir -p logs
nohup uv run kuaa serve --host 0.0.0.0 --port 8501 > logs/kuaa-web.log 2>&1 &

# O processo continua mesmo após logout. Para parar:
pkill -f "kuaa serve"
```

---

## 8. Executar pelo terminal (CLI)

O pacote também tem uma interface de linha de comando,
útil para processar vídeos em batch ou em servidores sem interface gráfica.

Crie a pasta `data/raw/` e coloque nela o vídeo que você quer processar.
Os exemplos abaixo usam `data/raw/<your-video>.mp4` como caminho de exemplo.

**Inspecionar um vídeo:**
```bash
uv run kuaa info data/raw/<your-video>.mp4
```

**Processar um vídeo completo:**
```bash
uv run kuaa process data/raw/<your-video>.mp4
```

**Escolher o identificador do filme** (`--slug`):
```bash
uv run kuaa process data/raw/<your-video>.mp4 --slug <slug>
```
Sem `--slug`, o identificador é o nome do arquivo, slugificado.
Os arquivos do filme ficam em `data/library/<slug>/` (veja
[Estrutura de dados gerada](#estrutura-de-dados-gerada)).

**Processar com configuração personalizada:**
```bash
uv run kuaa process data/raw/<your-video>.mp4 --config config/local.yaml
```

**Executar apenas etapas específicas** (útil para retomar processamento interrompido):
```bash
# Só detecção de cenas e embeddings (pula análise visual e descrição LLM)
uv run kuaa process data/raw/<your-video>.mp4 --steps scenes,embeddings
```

Nomes válidos para `--steps`:
`scenes`, `visual`, `embeddings`, `llm`
(separados por vírgula, sem espaços)

**Listar, ver a configuração e remover filmes:**
```bash
uv run kuaa library list                 # filmes registrados e seu estado
uv run kuaa config show                  # configuração efetiva (default + local)
uv run kuaa library delete <slug>        # remove o filme e seus arquivos (pede confirmação)
```

**Medir movimento por cena** (opcional, etapa separada):
```bash
uv run kuaa motion run
```

---

## 9. Primeira execução — o que esperar

Na primeira execução, o sistema baixa os modelos automaticamente.
Esse download exige acesso à internet. Isso acontece **uma vez** — ficam
em cache para as execuções seguintes. Depois do download, a inferência
roda offline.

| Modelo | Tamanho | Quando |
|---|---|---|
| YOLOv8n | ~6 MB | Primeiro uso da análise visual |
| SigLIP 2 (`google/siglip2-large-patch16-256`) | — | Primeiro uso de embeddings ou busca |
| Moondream 2 | ~1.9 GB | Primeiro uso de descrição LLM |

**Tempo estimado por etapa** (filme de ~90min, CPU):

| Etapa | Tempo aproximado |
|---|---|
| Detecção de cenas | 5–15 min |
| Análise visual | 10–20 min |
| Embeddings CLIP | 5–10 min |
| Descrição LLM | **60–120 min** (o mais demorado) |

O módulo LLM (Moondream) é o gargalo principal em CPU.
Com Apple Silicon M2 ou GPU NVIDIA, ele é 3–5× mais rápido.

**O processamento pode ser interrompido e retomado.**
O sistema salva checkpoints a cada 25 frames (`llm.checkpoint_interval`, padrão `25`). Se interromper
com Ctrl+C e executar novamente, ele continua de onde parou.

---

## 10. Solução de problemas comuns

**`ModuleNotFoundError: No module named 'kuaa'`**
```bash
# Ambiente virtual não está ativo:
source .venv/bin/activate

# Ou o pacote não está instalado:
pip install -e ".[full]"
```

**`RuntimeError: FFprobe não encontrado. Instale o FFmpeg: https://ffmpeg.org/download.html`**
```bash
# Verificar se estão instalados:
which ffmpeg ffprobe   # cada um deve retornar um caminho, não vazio

# Se não estiver, instalar (ver seção 5)
```

**`RuntimeError: FFprobe não encontrado`**
FFprobe vem junto com FFmpeg. Se FFmpeg está instalado mas FFprobe
não é encontrado, o PATH pode estar errado:
```bash
export PATH="/usr/local/bin:$PATH"   # macOS Homebrew
# Adicione esta linha ao ~/.bashrc ou ~/.zshrc para ser permanente
```

**`MTCNN falhou com MPS, usando CPU`**
Este é apenas um aviso — o sistema funciona normalmente em CPU.
Para desativar o aviso, edite `config/local.yaml`:
```yaml
hardware:
  device: "cpu"
```

**`torch.OutOfMemoryError` ou sistema travando**
Reduza o batch size no `config/local.yaml`:
```yaml
embeddings:
  batch_size: 4    # padrão é 16 — reduza se a memória for limitada
```

**Interface não abre no navegador**
```bash
# Acessar manualmente:
open http://localhost:8501   # macOS
xdg-open http://localhost:8501   # Linux

# Ou verificar se outra instância já está usando a porta:
lsof -i :8501
```

---

## 11. Atualizar para uma nova versão

**Se você clonou do GitHub:**
```bash
# Dentro da pasta do projeto:
git pull origin main
uv sync --extra full --group dev   # re-sincroniza caso pyproject.toml tenha mudado
```

**Verificar o que mudou antes de atualizar:**
```bash
git fetch origin
git log HEAD..origin/main --oneline   # lista os commits novos
git diff HEAD origin/main -- pyproject.toml   # ver mudanças de dependências
```

**Se uma atualização quebrar algo:**
```bash
# Voltar para a versão anterior (substitua o hash pelo commit desejado)
git log --oneline    # encontrar o hash do commit anterior
git checkout abc1234   # voltar para aquele estado
```

---

## Estrutura de dados gerada

Cada filme processado ocupa uma pasta `data/library/<slug>/`.
Esta é a estrutura que o sistema cria para um filme:

```
data/
├── raw/                            ← vídeos de entrada (você cria esta pasta)
│   └── <your-video>.mp4
└── library/
    ├── films.json                  ← registro dos filmes
    └── <slug>/
        ├── raw/
        ├── frames/
        │   └── scenes/
        │       └── keyframes_content/   ← keyframes por cena
        │           └── ...
        ├── embeddings/
        │   ├── keyframe_embeddings.npy  ← vetores SigLIP 2 (busca semântica)
        │   └── index_mapping.json       ← mapeamento índice → cena
        └── metadata/
            ├── scene_cuts.json          ← cortes detectados (editáveis na aba Pré-processamento)
            ├── keyframes_metadata.json  ← timecodes de cada cena
            ├── visual_analysis.json     ← faces, objetos, ambiente
            ├── scene_descriptions.json  ← descrições geradas pelo LLM
            ├── scene_tags.json          ← índice invertido de tags
            └── run_manifest.json        ← proveniência da última execução
```

Estes arquivos são a "memória" do sistema — uma vez gerados, a
interface de busca carrega instantaneamente sem reprocessar o vídeo.
