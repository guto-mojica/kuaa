"""English↔Portuguese lexicon for index-time BM25 expansion.

Why this exists
---------------
Every describer prompt in ``config/domains/*.yaml`` is written in English,
so Moondream returns English captions, and ``_generate_tags`` kebab-cases
the English ``objects`` / ``setting`` strings verbatim. Only four closed
vocabularies are translated (``LOCATION_MAP`` / ``TIME_MAP`` and the
people-count tags in ``kuaa.models.describer._common``).

The result is a BM25 corpus that is ~99% English while the interface, the
curators, and the queries are Brazilian Portuguese. Measured on
``jeca_tatu_1959`` (448 documents), the accented vocabulary of the entire
film is two words, and PT content queries return *literally zero* hits —
``homem a cavalo``, ``cavalo``, ``casa``, ``mulher de vestido`` all score
nothing, while their English equivalents return full result sets.

This module closes that gap without re-running the describer: at corpus
build time each English term contributes its Portuguese surface forms to
the same BM25 document, so a PT query has Portuguese to match against.
The expansion is purely additive — no English token is removed, so no
English query loses a hit.

Why a lexicon and not a translation model
------------------------------------------
Running models locally is a hard project constraint (no cloud APIs), and
an archivist asking "why did this match?" can read a table but not a
seq2seq model. The lexicon is also free at query time. A model-based pass
can replace :func:`pt_forms` later without touching its callers.

Scope
-----
Seeded from the actual corpus vocabulary — the ~260 most frequent terms
across the indexed library — not from a general dictionary. The long tail
of English is deliberately absent: terms that never appear in an archive
caption cost index size and buy nothing.

Plural forms are listed explicitly rather than stemmed, both because the
tokenizer's :func:`~kuaa.retrieval.tokenize.normalise_suffix` is
deliberately conservative and because an explicit table stays readable
when someone asks why a query matched.
"""

from __future__ import annotations

from kuaa.retrieval.tokenize import fold_diacritics, tokenize

# English term → Portuguese surface forms. Keys are lowercase and appear
# in both singular and plural where the corpus uses both. Values carry
# accents; callers fold them when the active tokenizer folds.
_EN_TO_PT: dict[str, tuple[str, ...]] = {
    # ── people ───────────────────────────────────────────────────────────
    "man": ("homem",),
    "men": ("homens",),
    "woman": ("mulher",),
    "women": ("mulheres",),
    "person": ("pessoa",),
    "people": ("pessoas", "gente"),
    "child": ("criança", "menino", "menina"),
    "children": ("crianças",),
    "boy": ("menino", "garoto"),
    "girl": ("menina", "garota"),
    "young": ("jovem",),
    "old": ("velho", "idoso"),
    "family": ("família",),
    "group": ("grupo",),
    "crowd": ("multidão",),
    "figure": ("figura", "vulto"),
    "cowboy": ("vaqueiro", "peão"),
    "farmer": ("agricultor", "lavrador", "roceiro"),
    "worker": ("trabalhador", "operário"),
    "actor": ("ator",),
    "actors": ("atores",),
    "couple": ("casal",),
    # ── body ─────────────────────────────────────────────────────────────
    "hand": ("mão",),
    "hands": ("mãos",),
    "head": ("cabeça",),
    "face": ("rosto", "cara"),
    "hair": ("cabelo",),
    "arm": ("braço",),
    "body": ("corpo",),
    "mustache": ("bigode",),
    "beard": ("barba",),
    # ── clothing ─────────────────────────────────────────────────────────
    "hat": ("chapéu",),
    "hats": ("chapéus",),
    "shirt": ("camisa",),
    "dress": ("vestido",),
    "suit": ("terno",),
    "jacket": ("jaqueta", "casaco"),
    "coat": ("casaco", "sobretudo"),
    "pants": ("calça", "calças"),
    "clothing": ("roupa", "vestimenta"),
    "clothes": ("roupas",),
    "attire": ("traje", "vestimenta"),
    "scarf": ("lenço", "cachecol"),
    "headscarf": ("lenço",),
    "glasses": ("óculos",),
    "shoe": ("sapato",),
    "shoes": ("sapatos",),
    "straw": ("palha",),
    "plaid": ("xadrez",),
    "striped": ("listrado",),
    "patterned": ("estampado",),
    # ── animals ──────────────────────────────────────────────────────────
    "horse": ("cavalo",),
    "horses": ("cavalos",),
    "cow": ("vaca", "boi"),
    "cows": ("vacas", "bois"),
    "ox": ("boi",),
    "oxen": ("bois",),
    "cattle": ("gado",),
    "dog": ("cachorro", "cão"),
    "dogs": ("cachorros", "cães"),
    "cat": ("gato",),
    "bird": ("pássaro", "ave"),
    "birds": ("pássaros", "aves"),
    "animal": ("animal",),
    "animals": ("animais",),
    "chicken": ("galinha",),
    "pig": ("porco",),
    # ── landscape / nature ───────────────────────────────────────────────
    "field": ("campo", "roça"),
    "fields": ("campos",),
    "rural": ("rural", "roceiro"),
    "farm": ("fazenda", "sítio", "roça"),
    "tree": ("árvore",),
    "trees": ("árvores",),
    "grass": ("grama", "capim"),
    "grassy": ("gramado",),
    "forest": ("floresta", "mata"),
    "wooded": ("arborizado",),
    "foliage": ("folhagem",),
    "plant": ("planta",),
    "plants": ("plantas",),
    "bush": ("arbusto",),
    "bushes": ("arbustos",),
    "garden": ("jardim", "horta"),
    "hill": ("colina", "morro"),
    "hills": ("colinas", "morros"),
    "mountain": ("montanha",),
    "mountains": ("montanhas", "serra"),
    "sky": ("céu",),
    "cloud": ("nuvem",),
    "clouds": ("nuvens",),
    "cloudy": ("nublado",),
    "water": ("água",),
    "river": ("rio",),
    "sea": ("mar",),
    "rock": ("pedra", "rocha"),
    "rocks": ("pedras", "rochas"),
    "rocky": ("rochoso", "pedregoso"),
    "cave": ("caverna", "gruta"),
    "dirt": ("terra", "barro"),
    "ground": ("chão", "solo"),
    "landscape": ("paisagem",),
    "sugarcane": ("cana",),
    "cane": ("cana",),
    "hay": ("feno", "palha"),
    # ── built environment ────────────────────────────────────────────────
    "house": ("casa",),
    "houses": ("casas",),
    "hut": ("cabana", "choupana", "casebre"),
    "building": ("prédio", "construção", "edifício"),
    "structure": ("estrutura", "construção"),
    "wall": ("parede", "muro"),
    "roof": ("telhado",),
    "thatched": ("sapé",),
    "door": ("porta",),
    "doorway": ("porta", "vão"),
    "window": ("janela",),
    "floor": ("chão", "piso"),
    "room": ("quarto", "sala", "cômodo"),
    "fence": ("cerca",),
    "gate": ("portão",),
    "street": ("rua",),
    "road": ("estrada", "rua"),
    "path": ("caminho", "trilha"),
    "sidewalk": ("calçada",),
    "urban": ("urbano",),
    "village": ("vila", "povoado"),
    "town": ("cidade", "vila"),
    "city": ("cidade",),
    "barn": ("celeiro",),
    "church": ("igreja",),
    "bar": ("bar", "boteco"),
    "shop": ("loja",),
    "market": ("mercado", "feira"),
    "park": ("parque",),
    "bridge": ("ponte",),
    "studio": ("estúdio",),
    "industrial": ("industrial",),
    "home": ("casa", "lar"),
    # ── objects ──────────────────────────────────────────────────────────
    "table": ("mesa",),
    "chair": ("cadeira",),
    "bench": ("banco",),
    "bed": ("cama",),
    "bottle": ("garrafa",),
    "bottles": ("garrafas",),
    "glass": ("copo", "vidro"),
    "cup": ("xícara", "copo"),
    "plate": ("prato",),
    "basket": ("cesta", "cesto"),
    "bag": ("bolsa", "saco"),
    "box": ("caixa",),
    "book": ("livro",),
    "paper": ("papel",),
    "car": ("carro", "automóvel"),
    "cars": ("carros",),
    "truck": ("caminhão",),
    "cart": ("carroça", "carro"),
    "wagon": ("carroça",),
    "wheel": ("roda",),
    "bicycle": ("bicicleta",),
    "motorcycle": ("moto", "motocicleta"),
    "train": ("trem",),
    "boat": ("barco",),
    "airplane": ("avião",),
    "stick": ("vara", "pau", "bengala"),
    "rope": ("corda",),
    "tool": ("ferramenta",),
    "tools": ("ferramentas",),
    "hoe": ("enxada",),
    "plow": ("arado",),
    "knife": ("faca",),
    "gun": ("arma", "revólver"),
    "pipe": ("cachimbo",),
    "cigarette": ("cigarro",),
    "smoke": ("fumaça",),
    "umbrella": ("guarda-chuva", "sombrinha"),
    "camera": ("câmera",),
    "microphone": ("microfone",),
    "bottleneck": ("gargalo",),
    "wooden": ("madeira", "de-madeira"),
    "wood": ("madeira",),
    "metal": ("metal",),
    "brick": ("tijolo",),
    "bamboo": ("bambu",),
    "stone": ("pedra",),
    "cloth": ("pano", "tecido"),
    "textured": ("texturizado",),
    # ── actions ──────────────────────────────────────────────────────────
    "walking": ("caminhando", "andando"),
    "walks": ("caminha", "anda"),
    "walk": ("caminhar", "andar"),
    "running": ("correndo",),
    "riding": ("cavalgando", "montando"),
    "rides": ("cavalga", "monta"),
    "standing": ("em-pé", "parado"),
    "stands": ("está-em-pé",),
    "sitting": ("sentado",),
    "sits": ("senta",),
    "seated": ("sentado",),
    "lying": ("deitado",),
    "lies": ("deitado",),
    "talking": ("conversando", "falando", "conversa"),
    "converse": ("conversar", "conversando", "conversa"),
    "conversing": ("conversando", "conversa"),
    "chatting": ("conversando", "batendo-papo"),
    "conversation": ("conversa", "conversando"),
    "holding": ("segurando",),
    "holds": ("segura",),
    "carrying": ("carregando",),
    "working": ("trabalhando",),
    "smoking": ("fumando",),
    "looking": ("olhando",),
    "observing": ("observando",),
    "gesturing": ("gesticulando",),
    "leaning": ("apoiado", "inclinado"),
    "leans": ("apoia",),
    "gathered": ("reunidos",),
    "surrounded": ("cercado", "rodeado"),
    "hanging": ("pendurado",),
    "eating": ("comendo",),
    "drinking": ("bebendo",),
    "dancing": ("dançando", "dança"),
    "playing": ("tocando", "brincando"),
    "singing": ("cantando",),
    "sleeping": ("dormindo",),
    # ── light / time / mood ──────────────────────────────────────────────
    "dark": ("escuro", "escura"),
    "darkness": ("escuridão",),
    "light": ("luz",),
    "lit": ("iluminado", "iluminada"),
    "bright": ("claro", "brilhante"),
    "dimly": ("mal-iluminado", "penumbra"),
    "shadow": ("sombra",),
    "shadows": ("sombras",),
    "shadowy": ("sombrio", "sombria", "sombreado"),
    "night": ("noite", "noturno", "noturna"),
    "day": ("dia", "diurno"),
    "daytime": ("dia",),
    "morning": ("manhã",),
    "evening": ("tarde", "entardecer"),
    "sunlight": ("sol", "luz-do-sol"),
    "dramatic": ("dramático",),
    "rustic": ("rústico",),
    "simple": ("simples", "humilde"),
    "empty": ("vazio",),
    "crowded": ("cheio", "lotado"),
    "quiet": ("silencioso", "calmo"),
    "natural": ("natural",),
    "rough": ("áspero", "rústico"),
    "nocturnal": ("noturno", "noturna"),
    "intense": ("intenso", "intensa", "intensas", "forte"),
    "low": ("baixo", "baixa"),
    "high": ("alto", "alta"),
    "alone": ("sozinho", "solitário", "solitária"),
    "solitary": ("solitário", "solitária"),
    "single": ("único", "única"),
    "lonely": ("solitário", "solitária"),
    # ── celebration / music ──────────────────────────────────────────────
    "party": ("festa",),
    "celebration": ("festa", "comemoração"),
    "festival": ("festa", "festival", "arraial"),
    "music": ("música",),
    "musician": ("músico",),
    "dance": ("dança",),
    "guitar": ("violão", "guitarra"),
    "accordion": ("sanfona", "acordeão"),
    "drum": ("tambor",),
    "popular": ("popular",),
    # ── colors ───────────────────────────────────────────────────────────
    "white": ("branco",),
    "black": ("preto", "negro"),
    "red": ("vermelho",),
    "blue": ("azul",),
    "green": ("verde",),
    "yellow": ("amarelo",),
    "brown": ("marrom",),
    "gray": ("cinza",),
    "grey": ("cinza",),
    "colored": ("colorido",),
    # ── framing / film ───────────────────────────────────────────────────
    "scene": ("cena",),
    "film": ("filme",),
    "image": ("imagem",),
    "frame": ("quadro", "enquadramento"),
    "foreground": ("primeiro-plano",),
    "background": ("fundo", "segundo-plano"),
    "backdrop": ("fundo",),
    "distance": ("distância", "ao-longe"),
    "interior": ("interior", "dentro"),
    "exterior": ("exterior", "fora"),
    "indoor": ("interior", "dentro"),
    "outdoor": ("exterior", "fora"),
    "outdoors": ("exterior", "ao-ar-livre"),
    "title": ("título",),
    "lettering": ("letras", "letreiro"),
    "text": ("texto",),
    "setting": ("cenário", "ambiente"),
    "environment": ("ambiente",),
    "area": ("área", "região"),
    "space": ("espaço",),
    "place": ("lugar",),
    "small": ("pequeno",),
    "large": ("grande",),
    "long": ("longo", "comprido"),
    "open": ("aberto",),
    "two": ("dois", "duas"),
    "three": ("três",),
    "one": ("um", "uma"),
}


def _build_inverse() -> dict[str, tuple[str, ...]]:
    """Invert :data:`_EN_TO_PT`, folding keys so accents are optional.

    Both the accented and unaccented spellings map to the same English
    terms, because a curator may type either.
    """
    inverse: dict[str, list[str]] = {}
    for english, pt_terms in _EN_TO_PT.items():
        for pt in pt_terms:
            for key in {pt, fold_diacritics(pt)}:
                bucket = inverse.setdefault(key, [])
                if english not in bucket:
                    bucket.append(english)
    return {k: tuple(v) for k, v in inverse.items()}


_PT_TO_EN: dict[str, tuple[str, ...]] = _build_inverse()


def pt_forms(token: str) -> tuple[str, ...]:
    """Portuguese surface forms for one English token. Empty when unknown."""
    return _EN_TO_PT.get(token, ())


def en_forms(token: str) -> tuple[str, ...]:
    """English terms for one Portuguese token, accented or not.

    Empty when unknown. Used on the query side, where a PT query token
    needs to reach English detector labels and captions.
    """
    return _PT_TO_EN.get(token, ())


def expand_text(text: str) -> str:
    """Return the Portuguese terms implied by the English in ``text``.

    Returns a space-joined string rather than a token list so callers can
    hand it straight to their own tokenizer — that way the PT forms pass
    through exactly the same folding and suffix normalisation as the rest
    of the document, instead of bypassing it.

    Order-stable and deduplicated, so a rebuilt index is byte-identical
    for unchanged input.
    """
    if not text:
        return ""
    seen: set[str] = set()
    out: list[str] = []
    for token in tokenize(text):
        for pt in pt_forms(token):
            if pt not in seen:
                seen.add(pt)
                out.append(pt)
    return " ".join(out)


def expand_query_tokens(tokens: list[str]) -> list[str]:
    """Append English equivalents to a Portuguese query token list.

    The mirror of :func:`expand_text`, for surfaces that are matched
    directly against English strings at query time rather than through a
    pre-built index — notably the YOLOv8 detected-object leg, whose
    labels are the 80 English COCO class names.

    Unknown tokens pass through unchanged, so an English query is
    untouched.
    """
    if not tokens:
        return tokens
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for candidate in (token, *en_forms(token)):
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out
