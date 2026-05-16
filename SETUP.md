# SETUP.md — Cinemateca AI

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
git clone https://github.com/cinemateca-brasileira/cinemateca-ai.git
cd cinemateca-ai
```

**Opção B — a partir de um arquivo .zip** (distribuição direta):
```bash
unzip cinemateca-ai.zip
cd cinemateca-ai
```

A estrutura que você deve ver:
```
cinemateca-ai/
├── app.py                  ← interface Streamlit
├── pyproject.toml          ← definição do pacote
├── config/
│   └── default.yaml        ← parâmetros padrão
├── src/
│   └── cinemateca/         ← módulos Python
└── tests/
    └── test_smoke.py
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
cd cinemateca-ai
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

# Sem uv (alternativa — funciona porque não há lockfile):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -e ".[full]" && pip install pytest pytest-cov black ruff mypy
```

O que `-e ".[full]"` significa:
- `-e` = "editable install" — o código em `src/` é usado diretamente,
  sem precisar reinstalar a cada alteração
- `[full]` = instalar todas as dependências opcionais (torch, CLIP,
  Streamlit, YOLOv8, etc.)

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
uv sync --extra gui       # só Streamlit
```

**Verificar instalação:**
```bash
uv run python -c "import cinemateca; print(cinemateca.__version__)"
# Deve imprimir: 0.1.0-alpha
```

---

## 5. Instalar FFmpeg

FFmpeg é usado para extração de frames e inspeção de vídeo.
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
  data_dir: "/dados/acervo/cinemateca"   # onde ficam os vídeos e frames
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
# Iniciar a interface Streamlit (usa config/default.yaml)
streamlit run app.py

# Ou com seu arquivo de configuração local:
streamlit run app.py
# e configure os caminhos dentro da interface
```

O Streamlit abrirá automaticamente no navegador padrão.
Se não abrir, acesse manualmente: **http://localhost:8501**

**Para rodar em uma máquina remota** (servidor da cinemateca) e
acessar do seu computador:
```bash
# Na máquina remota:
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# No seu navegador local:
# http://IP-DA-MAQUINA-REMOTA:8501
```

**Para manter rodando após fechar o terminal** (em servidores):
```bash
# Usando nohup (simples):
nohup streamlit run app.py --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &

# O processo continua mesmo após logout. Para parar:
pkill -f "streamlit run app.py"
```

---

## 8. Executar pelo terminal (CLI)

O pacote também tem uma interface de linha de comando,
útil para processar vídeos em batch ou em servidores sem interface gráfica.

**Inspecionar um vídeo:**
```bash
python -m cinemateca info --video data/raw/jeca_tatu_1959.mp4
```

**Processar um vídeo completo:**
```bash
python -m cinemateca process --video data/raw/jeca_tatu_1959.mp4
```

**Processar com configuração personalizada:**
```bash
python -m cinemateca process \
    --video data/raw/jeca_tatu_1959.mp4 \
    --config config/local.yaml
```

**Executar apenas etapas específicas** (útil para retomar processamento interrompido):
```bash
# Só detecção de cenas e embeddings (pula frames e análise visual)
python -m cinemateca process \
    --video data/raw/jeca_tatu_1959.mp4 \
    --steps scenes,embeddings
```

Nomes válidos para `--steps`:
`frames`, `scenes`, `visual`, `embeddings`, `llm`
(separados por vírgula, sem espaços)

---

## 9. Primeira execução — o que esperar

Na primeira execução, o sistema baixa os modelos automaticamente.
Isso acontece **uma vez** — ficam em cache para as execuções seguintes.

| Modelo | Tamanho | Quando |
|---|---|---|
| YOLOv8n | ~6 MB | Primeiro uso da análise visual |
| CLIP ViT-B/32 | ~150 MB | Primeiro uso de embeddings ou busca |
| Moondream 2 | ~1.9 GB | Primeiro uso de descrição LLM |

**Tempo estimado por etapa** (filme de ~90min, CPU):

| Etapa | Tempo aproximado |
|---|---|
| Extração de frames | 2–5 min |
| Detecção de cenas | 5–15 min |
| Análise visual | 10–20 min |
| Embeddings CLIP | 5–10 min |
| Descrição LLM | **60–120 min** (o mais demorado) |

O módulo LLM (Moondream) é o gargalo principal em CPU.
Com Apple Silicon M2 ou GPU NVIDIA, ele é 3–5× mais rápido.

**O processamento pode ser interrompido e retomado.**
O sistema salva checkpoints a cada 25 cenas. Se interromper
com Ctrl+C e executar novamente, ele continua de onde parou.

---

## 10. Solução de problemas comuns

**`ModuleNotFoundError: No module named 'cinemateca'`**
```bash
# Ambiente virtual não está ativo:
source .venv/bin/activate

# Ou o pacote não está instalado:
pip install -e ".[full]"
```

**`RuntimeError: FFmpeg não encontrado`**
```bash
# Verificar se está instalado:
which ffmpeg   # deve retornar um caminho, não vazio

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

**Streamlit travado em "Please wait..."**
Normalmente indica que um modelo está sendo baixado ou processamento
está em andamento. Verifique o terminal onde o Streamlit foi iniciado —
o log real de progresso aparece lá.

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

Após processar um vídeo, esta é a estrutura que o sistema cria:

```
data/
├── raw/
│   └── jeca_tatu_1959.mp4          ← vídeo original
├── frames/
│   ├── sample/                     ← frames extraídos (1/segundo)
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── scenes/
│       └── keyframes_content/      ← keyframes por cena
│           ├── Scene-001-01.jpg
│           └── ...
├── embeddings/
│   ├── keyframe_embeddings.npy     ← vetores CLIP (busca semântica)
│   └── index_mapping.json          ← mapeamento índice → cena
└── metadata/
    ├── video_properties.json       ← resolução, fps, duração
    ├── keyframes_metadata.json     ← timecodes de cada cena
    ├── visual_analysis.json        ← faces, objetos, ambiente
    ├── scene_descriptions.json     ← descrições geradas pelo LLM
    └── scene_tags.json             ← índice invertido de tags
```

Estes arquivos são a "memória" do sistema — uma vez gerados, a
interface de busca carrega instantaneamente sem reprocessar o vídeo.
