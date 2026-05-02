"""
app.py — Cinemateca AI  |  Interface Streamlit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Três abas:
    Processar  — executa o pipeline de catalogação em um vídeo
    Pesquisar  — busca semântica (texto ou imagem) no acervo indexado
    Catálogo   — navega e filtra todas as cenas catalogadas
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Garantir que src/ esteja no path quando rodado diretamente
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ─── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cinemateca AI",
    page_icon="🎞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_config(config_path: str | None = None):
    from cinemateca.config import load_config
    return load_config(config_path)


@st.cache_resource
def load_search_index(embeddings_dir: str):
    """Carrega embeddings e índice para busca. Cacheado por diretório."""
    from cinemateca.embeddings import CLIPEmbedder
    emb_dir = Path(embeddings_dir)
    emb_path = emb_dir / "keyframe_embeddings.npy"
    map_path = emb_dir / "index_mapping.json"
    if not emb_path.exists() or not map_path.exists():
        return None, None, None
    embeddings, mapping, kf_df = CLIPEmbedder.load(emb_path, map_path)
    embedder = CLIPEmbedder()
    return embeddings, kf_df, embedder


def _load_json(path: Path) -> list | dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_merged_tags(metadata_dir: Path) -> dict:
    """Carrega scene_tags.json e mescla com manual_annotations.json."""
    from cinemateca.annotator import load as load_annotations, merge_tag_index
    llm_tags = _load_json(metadata_dir / "scene_tags.json") or {}
    annotations = load_annotations(metadata_dir)
    return merge_tag_index(llm_tags, annotations)


def _render_keyframe_grid(rows, cols: int = 4):
    """Renderiza uma grade de keyframes a partir de uma lista de dicts com 'filepath'."""
    for i in range(0, len(rows), cols):
        grid = st.columns(cols)
        for j, row in enumerate(rows[i: i + cols]):
            fp = Path(row.get("filepath", ""))
            with grid[j]:
                if fp.exists():
                    st.image(str(fp), width=300)
                else:
                    st.caption("imagem não encontrada")
                scene_id = row.get("scene_id", row.get("rank", ""))
                sim = row.get("similarity")
                label = f"Cena {scene_id}"
                if sim is not None:
                    label += f"  —  {sim:.3f}"
                st.caption(label)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎞 Cinemateca AI")
    st.caption("v0.2.1")
    st.markdown("---")

    local_yaml = Path("config/local.yaml")
    config_path = str(local_yaml) if local_yaml.exists() else None
    cfg = get_config(config_path)

    st.subheader("Configuração")
    st.write(f"**Config:** `{'local.yaml' if config_path else 'default'}`")
    st.write(f"**Device:** `{cfg.hardware.device}`")
    st.write(f"**Data dir:** `{cfg.paths.data_dir}`")

    st.markdown("---")
    st.subheader("Sobre")
    st.write(
        "Sistema de catalogação audiovisual com IA para acervos cinematográficos. "
        "Roda completamente offline, sem envio de dados para servidores externos."
    )

# ─── Abas ─────────────────────────────────────────────────────────────────────

tab_process, tab_search, tab_catalog, tab_annotate = st.tabs(
    ["Processar", "Pesquisar", "Catálogo", "Anotar"]
)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — PROCESSAR
# ══════════════════════════════════════════════════════════════════════════════

with tab_process:
    st.header("Processar vídeo")
    st.write("Execute o pipeline completo de catalogação em um arquivo de vídeo.")

    col1, col2 = st.columns([2, 1])

    with col1:
        video_path_input = st.text_input(
            "Caminho do vídeo",
            placeholder="data/raw/jeca_tatu_1959.mp4",
            help="Caminho absoluto ou relativo ao diretório do projeto.",
        )

    with col2:
        custom_config = st.text_input(
            "Config personalizada (opcional)",
            placeholder="config/local.yaml",
            help="Deixe em branco para usar o default.",
        )

    st.subheader("Etapas")
    step_cols = st.columns(5)
    step_labels = {
        "frame_extraction": "Extração de frames",
        "scene_detection":  "Detecção de cenas",
        "visual_analysis":  "Análise visual",
        "embeddings":       "Embeddings CLIP",
        "llm_description":  "Descrição LLM",
    }
    steps_enabled = {}
    for i, (key, label) in enumerate(step_labels.items()):
        with step_cols[i]:
            default = getattr(cfg.pipeline.steps, key, True)
            steps_enabled[key] = st.checkbox(label, value=default, key=f"step_{key}")

    st.markdown("---")
    run_btn = st.button("Iniciar pipeline", type="primary", disabled=not video_path_input)

    if run_btn and video_path_input:
        video_path = Path(video_path_input)
        if not video_path.exists():
            st.error(f"Arquivo não encontrado: `{video_path}`")
        else:
            # Recarregar config com override do usuário se fornecido
            run_cfg = get_config(custom_config if custom_config else config_path)

            # Aplicar seleção de etapas
            for key, enabled in steps_enabled.items():
                setattr(run_cfg.pipeline.steps, key, enabled)

            from cinemateca.config import setup_logging
            from cinemateca.pipeline import CatalogPipeline

            setup_logging(run_cfg)
            pipeline = CatalogPipeline(run_cfg)

            progress = st.progress(0, text="Iniciando pipeline...")
            log_area = st.empty()
            step_names = list(step_labels.keys())
            n_steps = len(step_names)

            with st.spinner("Processando..."):
                result = pipeline.run(video_path)

            progress.progress(1.0, text="Concluído.")

            # Mostrar resultado
            st.subheader("Resultado")
            for i, step in enumerate(result.steps):
                if step.skipped:
                    st.info(f"⏭ **{step.name}** — pulado")
                elif step.success:
                    st.success(f"✓ **{step.name}** — {step.duration_s:.1f}s")
                else:
                    st.error(f"✗ **{step.name}** — {step.error}")

            total = result.total_duration_s
            if result.success:
                st.success(f"Pipeline concluído em {total:.1f}s")
            else:
                st.warning(f"Pipeline finalizado com erros em {total:.1f}s")

            # Invalidar cache de busca para forçar recarga
            load_search_index.clear()


# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — PESQUISAR
# ══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.header("Busca semântica")

    emb_dir = str(cfg.paths.embeddings_dir)
    embeddings, kf_df, embedder = load_search_index(emb_dir)

    if embeddings is None:
        st.warning(
            "Índice de busca não encontrado. "
            "Execute o pipeline com a etapa **Embeddings CLIP** ativada primeiro."
        )
    else:
        tag_index_search = _load_merged_tags(cfg.paths.metadata_dir)
        available_tags_search = sorted(tag_index_search.keys())
        has_tags = bool(available_tags_search)

        model_label = cfg.embeddings.model if hasattr(cfg, "embeddings") else "ViT-B-32"
        tag_info = f" · {len(tag_index_search)} tags" if has_tags else " · sem tags (rode etapa LLM)"
        st.caption(f"{len(kf_df)} cenas indexadas · {model_label}{tag_info}")

        search_mode = st.radio(
            "Modo de busca", ["Por texto", "Por imagem"], horizontal=True
        )
        top_k = st.slider("Resultados", min_value=4, max_value=24, value=8, step=4)

        if search_mode == "Por texto":
            query_text = st.text_input(
                "Busca",
                placeholder="two people talking outdoors at daytime",
                help="CLIP foi treinado predominantemente em inglês — descrições em inglês tendem a ter melhor recall.",
            )

            filter_tags_search: list[str] = []
            if has_tags:
                filter_tags_search = st.multiselect(
                    "Filtrar por tags antes da busca (opcional)",
                    available_tags_search,
                    help="Pré-filtra cenas pelas tags do LLM antes do ranking semântico — melhora precisão.",
                )

            if query_text:
                with st.spinner("Calculando similaridade..."):
                    from cinemateca.embeddings import SemanticSearch
                    searcher = SemanticSearch(embeddings, kf_df, embedder)
                    if filter_tags_search and tag_index_search:
                        results_df = searcher.combined(
                            query_text,
                            filter_tags=filter_tags_search,
                            tag_index=tag_index_search,
                            top_k=top_k,
                        )
                    else:
                        results_df = searcher.by_text(query_text, top_k=top_k)

                if results_df.empty:
                    st.info("Nenhum resultado.")
                else:
                    st.subheader("Resultados")
                    _render_keyframe_grid(results_df.to_dict("records"))

        else:  # Por imagem
            uploaded = st.file_uploader(
                "Imagem de referência", type=["jpg", "jpeg", "png"]
            )
            if uploaded:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = Path(tmp.name)

                col_img, col_res = st.columns([1, 3])
                with col_img:
                    st.image(str(tmp_path), caption="Referência", width=300)

                with st.spinner("Calculando similaridade..."):
                    from cinemateca.embeddings import SemanticSearch
                    searcher = SemanticSearch(embeddings, kf_df, embedder)
                    results_df = searcher.by_image(tmp_path, top_k=top_k)

                with col_res:
                    if results_df.empty:
                        st.info("Nenhum resultado.")
                    else:
                        st.subheader("Resultados")
                        _render_keyframe_grid(results_df.to_dict("records"))

                tmp_path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 — CATÁLOGO
# ══════════════════════════════════════════════════════════════════════════════

with tab_catalog:
    st.header("Catálogo de cenas")

    meta_dir = cfg.paths.metadata_dir
    kf_meta = _load_json(meta_dir / "keyframes_metadata.json")
    descriptions = _load_json(meta_dir / "scene_descriptions.json")
    tag_index = _load_merged_tags(meta_dir)
    visual_data = _load_json(meta_dir / "visual_analysis.json")

    if kf_meta is None:
        st.warning(
            "Metadados de cenas não encontrados. "
            "Execute o pipeline com a etapa **Detecção de cenas** primeiro."
        )
    else:
        # Montar lookup de descrições e análise visual por cena
        desc_by_scene: dict = {}
        if descriptions:
            for d in descriptions:
                desc_by_scene[str(d.get("scene_id", ""))] = d

        visual_by_scene: dict = {}
        if visual_data:
            for v in visual_data:
                visual_by_scene[str(v.get("scene_id", ""))] = v

        # ── Filtros ────────────────────────────────────────────────────────────
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        available_tags: list[str] = sorted(tag_index.keys()) if tag_index else []
        selected_tags: list[str] = []
        if available_tags:
            with filter_col1:
                selected_tags = st.multiselect("Filtrar por tags", available_tags)

        with filter_col2:
            search_term = st.text_input("Buscar na descrição", placeholder="rural, people...")

        with filter_col3:
            cols_count = st.select_slider("Colunas", options=[2, 3, 4, 5], value=4)

        # ── Filtrar cenas ──────────────────────────────────────────────────────
        scenes = kf_meta  # lista de dicts com scene_id, filepath, timecode, etc.

        if selected_tags and tag_index:
            valid_ids: set = set(tag_index.get(selected_tags[0], []))
            for tag in selected_tags[1:]:
                valid_ids &= set(tag_index.get(tag, []))
            scenes = [s for s in scenes if str(s.get("scene_id", "")) in valid_ids]

        if search_term:
            term_lower = search_term.lower()
            filtered = []
            for s in scenes:
                sid = str(s.get("scene_id", ""))
                desc = desc_by_scene.get(sid, {})
                text_blob = " ".join(str(v) for v in desc.values()).lower()
                if term_lower in text_blob:
                    filtered.append(s)
            scenes = filtered

        st.caption(f"{len(scenes)} cenas exibidas")
        st.markdown("---")

        # ── Grade de cenas ────────────────────────────────────────────────────
        if not scenes:
            st.info("Nenhuma cena corresponde aos filtros.")
        else:
            for i in range(0, len(scenes), cols_count):
                grid = st.columns(cols_count)
                for j, scene in enumerate(scenes[i: i + cols_count]):
                    fp = Path(scene.get("filepath", ""))
                    sid = str(scene.get("scene_id", ""))
                    with grid[j]:
                        if fp.exists():
                            st.image(str(fp), width=300)
                        else:
                            st.caption("sem imagem")

                        # Timecode
                        tc = scene.get("timecode_start") or scene.get("start_timecode", "")
                        st.caption(f"**Cena {sid}**  {tc}")

                        # Descrição LLM
                        desc = desc_by_scene.get(sid)
                        if desc:
                            setting = desc.get("setting", "")
                            location = desc.get("location", "")
                            n_people = desc.get("num_people")
                            parts = [p for p in [location, setting] if p]
                            summary = " · ".join(parts)
                            if n_people is not None and n_people >= 0:
                                summary += f" · {n_people} pessoa(s)"
                            if summary:
                                st.caption(summary)

                            tags_list = desc.get("tags", [])
                            if tags_list:
                                st.caption(" ".join(f"`{t}`" for t in tags_list[:6]))

                        # Análise visual
                        vis = visual_by_scene.get(sid)
                        if vis:
                            faces = vis.get("num_faces", 0)
                            env = vis.get("environment", {})
                            loc = env.get("location", "")
                            time_of_day = env.get("time_of_day", "")
                            vis_parts = [p for p in [loc, time_of_day] if p]
                            if faces:
                                vis_parts.append(f"{faces} rosto(s)")
                            if vis_parts:
                                st.caption(" · ".join(vis_parts))


# ══════════════════════════════════════════════════════════════════════════════
# ABA 4 — ANOTAR
# ══════════════════════════════════════════════════════════════════════════════

with tab_annotate:
    st.header("Anotar cenas")
    st.write("Adicione tags manualmente a cenas sem descrição LLM ou com metadados incompletos.")

    from cinemateca.annotator import load as _load_annotations, save as _save_annotations

    _meta_dir = cfg.paths.metadata_dir
    _kf_meta  = _load_json(_meta_dir / "keyframes_metadata.json")
    _descs    = _load_json(_meta_dir / "scene_descriptions.json") or []

    if not _kf_meta:
        st.warning("Execute a etapa **Detecção de cenas** primeiro.")
    else:
        # IDs com descrição LLM válida (sem template quebrado e sem erro)
        _BROKEN = "One or two sentences about subject"
        _valid_desc_ids = {
            d["scene_id"] for d in _descs
            if "error" not in d and _BROKEN not in d.get("description", "")
        }

        _filter = st.radio(
            "Exibir",
            ["Sem descrição LLM", "Todas as cenas"],
            horizontal=True,
        )
        _scenes = (
            [s for s in _kf_meta if s["scene_id"] not in _valid_desc_ids]
            if _filter == "Sem descrição LLM"
            else _kf_meta
        )

        # Carregar anotações manuais (sem cache — leitura direta para refletir saves)
        _annotations = _load_annotations(_meta_dir)
        _annotated_n = sum(1 for s in _scenes if str(s["scene_id"]) in _annotations)

        st.caption(f"{len(_scenes)} cenas · {_annotated_n} já anotadas manualmente")
        st.markdown("---")

        if not _scenes:
            st.success("Todas as cenas têm descrição LLM válida.")
        else:
            # ── Navegação ──────────────────────────────────────────────────────
            if "annotate_idx" not in st.session_state:
                st.session_state.annotate_idx = 0
            # Clamp index in case the scene list changed
            st.session_state.annotate_idx = min(
                st.session_state.annotate_idx, len(_scenes) - 1
            )

            nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1])
            with nav_col1:
                if st.button("←", disabled=st.session_state.annotate_idx == 0):
                    st.session_state.annotate_idx -= 1
                    st.rerun()
            with nav_col3:
                if st.button("→", disabled=st.session_state.annotate_idx == len(_scenes) - 1):
                    st.session_state.annotate_idx += 1
                    st.rerun()
            with nav_col2:
                chosen = st.selectbox(
                    "Cena",
                    range(len(_scenes)),
                    format_func=lambda i: f"Cena {_scenes[i]['scene_id']}",
                    index=st.session_state.annotate_idx,
                    label_visibility="collapsed",
                )
                if chosen != st.session_state.annotate_idx:
                    st.session_state.annotate_idx = chosen

            _scene = _scenes[st.session_state.annotate_idx]
            _sid   = str(_scene["scene_id"])
            _fp    = Path(_scene.get("filepath", ""))

            # ── Conteúdo da cena ───────────────────────────────────────────────
            img_col, form_col = st.columns([1, 1])

            with img_col:
                if _fp.exists():
                    st.image(str(_fp))
                else:
                    st.caption("imagem não encontrada")
                _start = _scene.get("start_time_s", 0)
                _end   = _scene.get("end_time_s", 0)
                st.caption(f"⏱ {_start:.1f}s → {_end:.1f}s  ·  duração {_end - _start:.1f}s")

            with form_col:
                # LLM info (se existir)
                _llm = next((d for d in _descs if d["scene_id"] == _scene["scene_id"]), None)
                if _llm and _BROKEN not in _llm.get("description", ""):
                    st.markdown("**Descrição LLM**")
                    st.caption(_llm.get("description", ""))
                    _ltags = _llm.get("tags", [])
                    if _ltags:
                        st.caption(" ".join(f"`{t}`" for t in _ltags))
                else:
                    st.caption("_Sem descrição LLM_")

                st.markdown("---")
                st.markdown("**Tags manuais**")

                _existing = _annotations.get(_sid, [])
                _tag_input = st.text_input(
                    "Tags (separadas por vírgula)",
                    value=", ".join(_existing),
                    key=f"tag_input_{_sid}",
                    placeholder="rural, exterior, cavalo, pessoa-unica",
                    label_visibility="collapsed",
                )
                st.caption("Use hífens para tags compostas: `duas-pessoas`, `cena-noturna`")

                save_col, clear_col = st.columns(2)
                with save_col:
                    if st.button("Salvar", type="primary", key=f"save_{_sid}"):
                        _new_tags = [
                            t.strip().lower().replace(" ", "-")
                            for t in _tag_input.split(",") if t.strip()
                        ]
                        _annotations[_sid] = _new_tags
                        _save_annotations(_meta_dir, _annotations)
                        st.success(f"✓ {len(_new_tags)} tag(s) salvas")
                with clear_col:
                    if st.button("Limpar", key=f"clear_{_sid}"):
                        _annotations.pop(_sid, None)
                        _save_annotations(_meta_dir, _annotations)
                        st.session_state.pop(f"tag_input_{_sid}", None)
