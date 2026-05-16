"""Service layer for the FastAPI web app.

Services own the catalog/annotation/search domain logic that used to be
duplicated across ``api/routes/*``. Route functions stay thin: request
parsing + template rendering only. AI/model logic still lives in
``src/cinemateca`` — services orchestrate it, they do not reimplement it.

Phase 3a introduces ``catalog`` (scene/card/metadata/keyframe-url +
tag-index merge/normalization) and ``FilmContext`` (resolved artifact
paths). Phase 3b/3c add the annotations/search services on top of the
shared primitives exposed here.
"""
