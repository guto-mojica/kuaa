"""CMX 3600 EDL export from a selection of scene timecodes.

Returns plain text — no HTTP concerns. The route layer owns headers.
"""

from __future__ import annotations

from dataclasses import dataclass

from kuaa.library import to_smpte


@dataclass
class SceneSlice:
    scene_id: int
    film_slug: str
    film_title: str
    slug: str
    start_time_s: float
    end_time_s: float
    fps: float


def scenes_to_edl(scenes: list[SceneSlice], title: str = "Export") -> str:
    """Render a CMX 3600 EDL from an ordered list of scene slices.

    Source timecodes reflect the original film position.
    Record timecodes are assembled sequentially from 00:00:00:00.
    """
    lines: list[str] = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]

    record_s = 0.0
    for i, sc in enumerate(scenes, 1):
        fps = max(1.0, sc.fps)
        duration = max(0.0, sc.end_time_s - sc.start_time_s)

        src_in = to_smpte(sc.start_time_s, fps)
        src_out = to_smpte(sc.end_time_s, fps)
        rec_in = to_smpte(record_s, fps)
        record_s += duration
        rec_out = to_smpte(record_s, fps)

        # Reel name: film slug, max 8 chars, uppercase, alphanumeric only.
        reel = "".join(c for c in sc.film_slug.upper() if c.isalnum())[:8] or "AX"

        lines.append(f"{i:03d}  {reel:<8} V     C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {sc.slug}")
        if sc.film_title and sc.film_title != sc.film_slug:
            lines.append(f"* SOURCE FILE: {sc.film_title}")
        lines.append("")

    return "\n".join(lines)
