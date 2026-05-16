# Changelog

Todas as mudanças notáveis neste projeto serão documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).
Versionamento segue [Semantic Versioning](https://semver.org/lang/pt-BR/):
`MAJOR.MINOR.PATCH` — sufixo `-alpha` ou `-beta` indica versões de pré-lançamento.

---

## [Não lançado]

Funcionalidades planejadas para as próximas versões:

- Exportação de catálogo em CSV e JSON estruturado
- Interface para comparar resultados de diferentes thresholds de detecção
- Suporte a múltiplos vídeos no mesmo acervo indexado
- Detector de ambiente baseado em modelo treinado (substituir heurística atual)
- Docker image para instalação sem dependências manuais
- Testes de integração com vídeo de referência

### Ferramentas de desenvolvimento

- Adoção do **uv** para gerenciamento de ambiente e dependências.
  Dependências de desenvolvimento movidas para `[dependency-groups]`
  (PEP 735). Backend de build (`setuptools`) inalterado. Lockfile
  (`uv.lock`) adiado — instalação via `pip` continua funcionando.

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
