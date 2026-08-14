"""
kuaa.models.describer._common
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model-agnostic prompt set + answer parsing + tag/metadata assembly,
extracted unchanged from the former llm_describer.py. Shared by every
SceneDescriber backend so parsing/tagging behaviour is identical
regardless of the VLM engine.
"""

from __future__ import annotations

import re

import pandas as pd

from kuaa.errors import is_degenerate_response, is_unusable_response

PROMPTS: dict[str, tuple[str, int]] = {
    #                prompt                                          max_new_tokens
    "description": (
        "Describe this film scene in one or two sentences. "
        "Focus on the main subject, action, and setting.",
        80,
    ),
    "location": (
        "Is this scene indoors or outdoors? Answer with one word: indoor or outdoor.",
        10,
    ),
    "setting": (
        "Describe the setting in 2-4 words. Examples: urban street, rural field, farm, village.",
        20,
    ),
    "time_of_day": (
        "What time of day is this scene? Answer with one word: day, night, or unknown.",
        10,
    ),
    "people_and_action": (
        "How many people are visible and what are they doing? "
        "Answer briefly, e.g.: 2 people talking.",
        30,
    ),
    "objects": (
        "List the most notable objects in this scene, comma-separated. Maximum 6 items.",
        40,
    ),
}


#: Consecutive degenerate scenes tolerated before a describe run is aborted.
#: One bad frame (a title card, a black frame) can plausibly produce one
#: strange answer; three in a row means the backend stopped working, not that
#: the footage got harder. Shared by every backend so the abort policy does
#: not drift between them. Overridable via ``llm.max_consecutive_degenerate``.
DEFAULT_MAX_CONSECUTIVE_DEGENERATE = 3

#: Operator-facing explanation attached to the abort error.
DEGENERACY_ABORT_HINT = (
    "The describer is emitting repetition loops instead of answers. On Apple "
    "MPS this is what device memory exhaustion looks like — command buffers "
    "return garbage before they return an error. Restart the process (a "
    "long-lived server accumulates model weights across films; the accelerator "
    "allocator does not return them to the OS) and describe one film per run."
)


LOCATION_MAP = {
    "indoor": "interior",
    "indoors": "interior",
    "inside": "interior",
    "interior": "interior",
    "internal": "interior",
    "outdoor": "exterior",
    "outdoors": "exterior",
    "outside": "exterior",
    "exterior": "exterior",
    "external": "exterior",
}

TIME_MAP = {
    "day": "dia",
    "daytime": "dia",
    "daylight": "dia",
    "night": "noite",
    "nighttime": "noite",
    "dark": "noite",
    "unknown": "desconhecido",
    "unclear": "desconhecido",
}


def _parse_num_people(text: str) -> int:
    """
    Extrai número de pessoas de uma resposta em linguagem natural.

    Retorna:
        int >= 0 : número conhecido
        -1       : múltiplos/vago ("several", "many")
    """
    text = text.lower()
    if any(w in text for w in ["no people", "nobody", "no person", "empty", "no one"]):
        return 0

    word_to_num = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "a person": 1,
        "a man": 1,
        "a woman": 1,
        "one person": 1,
    }
    for word, num in word_to_num.items():
        if word in text:
            return num

    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))

    if any(w in text for w in ["several", "many", "group", "crowd", "multiple"]):
        return -1

    return -1


def _parse_objects(text: str) -> list[str]:
    """Converte a string de objetos em lista normalizada (máx. 6 itens)."""
    if not text or is_unusable_response(text):
        return []
    parts = re.split(r"[,;.]+", text)
    stopwords = {"a", "an", "the", "some", "and", "with"}
    objects = []
    for p in parts:
        words = [w for w in p.strip().lower().split() if w not in stopwords]
        cleaned = " ".join(words)
        if cleaned:
            objects.append(cleaned)
    return objects[:6]


def _generate_tags(parsed: dict) -> list[str]:
    """Gera lista de tags kebab-case a partir dos campos parseados."""
    tags = set()

    if parsed.get("location") in ("interior", "exterior"):
        tags.add(parsed["location"])

    if parsed.get("time_of_day") in ("dia", "noite"):
        tags.add(parsed["time_of_day"])

    n = parsed.get("num_people", -1)
    if n == 0:
        tags.add("sem-pessoas")
    elif n == 1:
        tags.add("pessoa-unica")
    elif n == 2:
        tags.add("duas-pessoas")
    elif n >= 3:
        tags.add("multiplas-pessoas")

    for obj in parsed.get("objects", []):
        tags.add(obj.replace(" ", "-").lower())

    setting = parsed.get("setting", "").strip().lower()
    if setting:
        tags.add(setting.replace(" ", "-"))

    return sorted(tags)


def _scene_provenance(row: pd.Series) -> dict:
    """Scene-level fields, carried by good and error rows alike."""
    return {
        "scene_id": int(row.get("scene_id", -1)),
        "keyframe_id": str(row.get("keyframe_id", "")),
        "keyframe_path": str(row.get("filepath", "")),
        "start_time_s": float(row["start_time_s"]) if "start_time_s" in row.index else None,
        "end_time_s": float(row["end_time_s"]) if "end_time_s" in row.index else None,
        "duration_s": float(row["duration_s"]) if "duration_s" in row.index else None,
    }


def degenerate_fields(raw: dict) -> list[str]:
    """Names of the prompt fields in *raw* that came back as repetition loops."""
    return sorted(f for f, text in raw.items() if is_degenerate_response(str(text)))


def build_metadata(row: pd.Series, raw: dict) -> dict:
    """Assemble one scene record from the raw per-prompt answers.

    Returns an **error row** (``error`` set, no ``description``, empty
    ``tags``) when any prompt came back as a repetition loop. The trigger is
    deliberately one field, not all six: degeneracy at these thresholds means
    the decoder was pinned by a device-level fault, so the answers that still
    look plausible are not trustworthy either. Dropping the scene is
    recoverable — ``describe_batch`` reprocesses error rows on resume —
    whereas indexing it silently degrades every future query.
    """
    bad = degenerate_fields(raw)
    if bad:
        return {
            **_scene_provenance(row),
            "error": (
                f"degenerate model output in {', '.join(bad)} — repetition loop, "
                "not an answer (see kuaa.errors.is_degenerate_response)"
            ),
            "tags": [],
            "objects": [],
            "_raw_responses": raw,
        }

    loc_raw = raw.get("location", "").lower().strip()
    location = LOCATION_MAP.get(loc_raw, "desconhecido")

    time_raw = raw.get("time_of_day", "").lower().strip()
    time_of_day = TIME_MAP.get(time_raw, "desconhecido")

    num_people = _parse_num_people(raw.get("people_and_action", ""))
    objects = _parse_objects(raw.get("objects", ""))
    setting = raw.get("setting", "").strip().lower()
    if is_unusable_response(setting):
        # ``setting`` is kebab-cased straight into the tag vocabulary by
        # ``_generate_tags``, with no other filter between it and the index.
        setting = ""

    parsed = {
        "location": location,
        "time_of_day": time_of_day,
        "num_people": num_people,
        "objects": objects,
        "setting": setting,
    }
    tags = _generate_tags(parsed)

    scene_meta = _scene_provenance(row)

    llm_meta = {
        "description": raw.get("description", "").strip(),
        "location": location,
        "setting": setting,
        "time_of_day": time_of_day,
        "num_people": num_people,
        "people_action": raw.get("people_and_action", "").strip(),
        "objects": objects,
        "tags": tags,
        "_raw_responses": raw,  # útil para depuração e auditoria
    }

    return {**scene_meta, **llm_meta}
