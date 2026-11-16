"""The pt_BR catalogue must be fully translated: fuzzy and empty entries show English."""

from __future__ import annotations

import re
from pathlib import Path

from babel.messages.catalog import Message
from babel.messages.pofile import read_po

PO = Path(__file__).resolve().parents[1] / "web/locales/pt_BR/LC_MESSAGES/messages.po"
PLACEHOLDER = re.compile(r"%\([A-Za-z_]\w*\)[sdif]|%[sdif]")


def _active() -> list[Message]:
    with PO.open("rb") as fh:
        return [m for m in read_po(fh) if m.id]


def _forms(m: Message) -> list[str]:
    return [m.string] if isinstance(m.string, str) else list(m.string)


def _label(m: Message) -> str:
    ids = m.id if isinstance(m.id, str) else m.id[0]
    loc = m.locations[0][0] if m.locations else "?"
    return f"{ids!r} ({loc})"


def test_no_fuzzy_entries() -> None:
    bad = [_label(m) for m in _active() if m.fuzzy]
    assert not bad, "fuzzy entries:\n" + "\n".join(bad)


def test_no_empty_msgstr() -> None:
    bad = [_label(m) for m in _active() if not all(_forms(m))]
    assert not bad, "empty msgstr:\n" + "\n".join(bad)


def test_placeholders_preserved() -> None:
    bad = []
    for m in _active():
        ids = [m.id] if isinstance(m.id, str) else list(m.id)
        want = {p for i in ids for p in PLACEHOLDER.findall(i)}
        for form in _forms(m):
            if form and not want <= set(PLACEHOLDER.findall(form)):
                bad.append(_label(m))
    assert not bad, "placeholder mismatch:\n" + "\n".join(bad)


def test_no_pesquisar() -> None:
    bad = [_label(m) for m in _active() if any("pesquisar" in f.lower() for f in _forms(m))]
    assert not bad, "use 'buscar', not 'pesquisar':\n" + "\n".join(bad)
