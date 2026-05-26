"""Audio-only retrieval — CLAP joint text+audio space.

Mirrors the BM25 loader pattern (mtime+size cache) and the CLIP search
service shape. Keeps the route layer thin: the route asks for a top-k
list of scored scenes; the loader + searcher live here.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from cinemateca.models.base import AudioEmbedder

logger = logging.getLogger(__name__)

_CLAP_EMB_NAME = "clap_embeddings.npy"
_CLAP_MAP_NAME = "audio_mapping.json"


class AudioMappingRow(TypedDict):
    """One row of the parallel ``AudioIndex.mapping`` list.

    ``scene_id`` is the integer scene id (the join key against the rest
    of the pipeline). ``wav_path`` is the relative path under the film
    dir. ``start_time_s`` / ``end_time_s`` are scene boundaries in
    seconds; ``None`` when the writer didn't supply them.
    """

    scene_id: int
    wav_path: str
    start_time_s: float | None
    end_time_s: float | None


@dataclass(frozen=True)
class AudioIndex:
    """In-memory CLAP index for one film.

    ``embeddings`` is (N, D) float32 L2-normalised — same convention as the
    on-disk write in ``ClapHFEmbedder.save``. ``mapping`` is the parallel
    JSON list (parallel arrays, same N).
    """

    embeddings: np.ndarray
    mapping: list[AudioMappingRow]


# Module-level cache, mtime+size-keyed (matches api/services/search.py's
# CLIP index loader). Single-worker dev server: a simple lock is enough.
_CACHE: dict[Path, tuple[tuple[int, int, int, int], AudioIndex]] = {}
_CACHE_LOCK = threading.Lock()


def _stat_key(emb_path: Path, map_path: Path) -> tuple[int, int, int, int] | None:
    try:
        es = emb_path.stat()
        ms = map_path.stat()
    except FileNotFoundError:
        return None
    return (es.st_mtime_ns, es.st_size, ms.st_mtime_ns, ms.st_size)


def load_audio_index(audio_dir: Path) -> AudioIndex | None:
    """Load (or return cached) CLAP index for one film's ``audio/`` dir.

    Returns ``None`` when either file is missing — audio is opt-in per
    the CLAP plan, so a film without audio embeddings is a normal state,
    not an error.
    """
    emb_path = audio_dir / _CLAP_EMB_NAME
    map_path = audio_dir / _CLAP_MAP_NAME
    if not emb_path.exists() or not map_path.exists():
        return None
    key = _stat_key(emb_path, map_path)
    if key is None:
        return None
    with _CACHE_LOCK:
        cached = _CACHE.get(audio_dir)
        if cached is not None and cached[0] == key:
            return cached[1]
        embeddings = np.load(emb_path).astype("float32", copy=False)
        mapping_raw = json.loads(map_path.read_text())
        # Normalise to list[dict]. The real ClapHFEmbedder.save() writes a
        # dict-of-parallel-arrays; older / synthetic writers may emit a
        # list-of-dicts directly. Both shapes map to the same row-aligned
        # AudioIndex.mapping contract.
        mapping: list[AudioMappingRow]
        if isinstance(mapping_raw, dict) and "scene_ids" in mapping_raw:
            sids = mapping_raw["scene_ids"]
            wavs = mapping_raw.get("wav_paths") or [""] * len(sids)
            starts = mapping_raw.get("start_times_s") or [None] * len(sids)
            ends = mapping_raw.get("end_times_s") or [None] * len(sids)
            mapping = [
                AudioMappingRow(
                    scene_id=int(sids[i]),
                    wav_path=str(wavs[i]),
                    start_time_s=(float(starts[i]) if starts[i] is not None else None),
                    end_time_s=(float(ends[i]) if ends[i] is not None else None),
                )
                for i in range(len(sids))
            ]
        elif isinstance(mapping_raw, list):
            # already row-shaped; widen to AudioMappingRow (coerce types defensively)
            mapping = [
                AudioMappingRow(
                    scene_id=int(m["scene_id"]),
                    wav_path=str(m.get("wav_path", "")),
                    start_time_s=(
                        float(m["start_time_s"]) if m.get("start_time_s") is not None else None
                    ),
                    end_time_s=(
                        float(m["end_time_s"]) if m.get("end_time_s") is not None else None
                    ),
                )
                for m in mapping_raw
            ]
        else:
            raise ValueError(
                f"Unrecognised CLAP mapping shape at {map_path}: "
                f"expected dict with 'scene_ids' or list of dicts."
            )
        if len(mapping) != embeddings.shape[0]:
            raise ValueError(
                f"CLAP index row count mismatch at {audio_dir}: "
                f"embeddings={embeddings.shape[0]} mapping={len(mapping)}"
            )
        # On-disk vectors are L2-normalised at write time; re-normalise
        # defensively in case the file was hand-edited or truncated.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        embeddings = (embeddings / norms).astype("float32")
        idx = AudioIndex(embeddings=embeddings, mapping=mapping)
        _CACHE[audio_dir] = (key, idx)
        return idx


def search_audio(
    index: AudioIndex,
    embedder: AudioEmbedder,
    query_text: str,
    *,
    top_k: int = 10,
) -> list[dict]:
    """Cosine-similarity search over CLAP embeddings.

    Returns a list of dicts ``{"scene_id": int, "score": float}`` ordered
    by descending score. ``query_text`` is encoded via
    ``embedder.encode_text`` (L2-normalised — CLAP backend guarantees
    this). Cosine reduces to a dot product because both sides are
    pre-normalised.
    """
    if not query_text.strip():
        return []
    q = embedder.encode_text(query_text)
    if q.ndim != 1 or q.shape[0] != index.embeddings.shape[1]:
        raise ValueError(
            f"Query vector dim {q.shape} incompatible with index dim "
            f"{index.embeddings.shape[1]}"
        )
    scores = index.embeddings @ q  # (N,) cosines
    k = min(int(top_k), scores.shape[0])
    # np.argpartition is O(N) and faster than argsort for k << N.
    if k <= 0:
        return []
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    out: list[dict] = []
    for i in top_idx:
        m = index.mapping[int(i)]
        out.append({"scene_id": int(m["scene_id"]), "score": float(scores[int(i)])})
    return out
