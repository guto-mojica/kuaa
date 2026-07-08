"""Vector-index backends (the retrieval-scaling seam).

A :class:`~kuaa.retrieval.vector_index.base.VectorIndex` abstracts the
nearest-neighbour lookup over CLIP/SigLIP keyframe embeddings. The default
``numpy_bruteforce`` backend reproduces the historical in-memory
``embeddings @ query`` + ``argsort`` exactly (byte-identical top-k); the
opt-in ``lancedb`` backend stores one on-disk table with a ``film_slug``
column so cross-film aggregate search collapses to a single filtered query.

Consumers reach a backend through :func:`get_vector_index`; the Protocol
lives in ``base`` and concrete backends in their own modules, mirroring the
model-backend registry pattern in ``kuaa.models``.
"""

from kuaa.retrieval.vector_index.base import VectorIndex
from kuaa.retrieval.vector_index.registry import get_vector_index

__all__ = ["VectorIndex", "get_vector_index"]
