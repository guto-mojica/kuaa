"""Frame-source backends (the input-side seam).

Concrete implementations of the :class:`kuaa.models.base.FrameSource`
Protocol. The pipeline reaches these only through
:func:`kuaa.models.registry.get_frame_source` — never by importing a
concrete backend directly — mirroring the model-backend registry pattern.
"""
