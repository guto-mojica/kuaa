"""Portuguese aliases for the English COCO class names YOLOv8 emits.

The object detector labels scenes with the 80 English COCO classes, but the
interface language is pt-BR, so a curator searching ``cavalo`` scores nothing
against a scene the detector labelled ``horse``. This map is consulted only when
matching a query against detector output — descriptions and manual tags are
already Portuguese and are matched verbatim.

Aliases are folded per-token: an alias whose English class is multi-word
(``potted plant``) expands to that class's tokens, so ``vaso`` matches it. Only
classes a film archive plausibly surfaces are aliased; the rest of COCO
(``frisbee``, ``snowboard``) is left to match on its English name.
"""

from __future__ import annotations

from kuaa.retrieval.tokenize import tokenize

# Portuguese token -> English COCO class name. Singular and plural forms are
# listed separately rather than stemmed: the tokenizer does not stem, and an
# explicit table stays readable when an archivist asks why a query matched.
_PT_TO_COCO: dict[str, str] = {
    # people
    "pessoa": "person",
    "pessoas": "person",
    "homem": "person",
    "mulher": "person",
    "gente": "person",
    "figura": "person",
    # vehicles
    "bicicleta": "bicycle",
    "bicicletas": "bicycle",
    "carro": "car",
    "carros": "car",
    "automovel": "car",
    "automóvel": "car",
    "moto": "motorcycle",
    "motocicleta": "motorcycle",
    "aviao": "airplane",
    "avião": "airplane",
    "onibus": "bus",
    "ônibus": "bus",
    "trem": "train",
    "caminhao": "truck",
    "caminhão": "truck",
    "barco": "boat",
    "barcos": "boat",
    # street furniture
    "semaforo": "traffic light",
    "semáforo": "traffic light",
    "hidrante": "fire hydrant",
    "parquimetro": "parking meter",
    "parquímetro": "parking meter",
    "banco": "bench",
    "placa": "stop sign",
    # animals
    "passaro": "bird",
    "pássaro": "bird",
    "ave": "bird",
    "gato": "cat",
    "gatos": "cat",
    "cachorro": "dog",
    "cachorros": "dog",
    "cao": "dog",
    "cão": "dog",
    "cavalo": "horse",
    "cavalos": "horse",
    "ovelha": "sheep",
    "ovelhas": "sheep",
    "vaca": "cow",
    "vacas": "cow",
    "boi": "cow",
    "gado": "cow",
    "elefante": "elephant",
    "urso": "bear",
    "zebra": "zebra",
    "girafa": "giraffe",
    # carried objects
    "mochila": "backpack",
    "sombrinha": "umbrella",
    "bolsa": "handbag",
    "gravata": "tie",
    "mala": "suitcase",
    "malas": "suitcase",
    # sport
    "bola": "sports ball",
    "pipa": "kite",
    "taco": "baseball bat",
    "skate": "skateboard",
    "prancha": "surfboard",
    "raquete": "tennis racket",
    "esquis": "skis",
    "esquís": "skis",
    # tableware and food
    "garrafa": "bottle",
    "garrafas": "bottle",
    "taca": "wine glass",
    "taça": "wine glass",
    "copo": "cup",
    "xicara": "cup",
    "xícara": "cup",
    "garfo": "fork",
    "faca": "knife",
    "colher": "spoon",
    "tigela": "bowl",
    "banana": "banana",
    "maca": "apple",
    "maçã": "apple",
    "sanduiche": "sandwich",
    "sanduíche": "sandwich",
    "laranja": "orange",
    "brocolis": "broccoli",
    "brócolis": "broccoli",
    "cenoura": "carrot",
    "pizza": "pizza",
    "rosquinha": "donut",
    "bolo": "cake",
    # furniture
    "cadeira": "chair",
    "cadeiras": "chair",
    "sofa": "couch",
    "sofá": "couch",
    "vaso": "vase",
    "planta": "potted plant",
    "cama": "bed",
    "mesa": "dining table",
    "privada": "toilet",
    # appliances and electronics
    "televisao": "tv",
    "televisão": "tv",
    "tv": "tv",
    "laptop": "laptop",
    "computador": "laptop",
    "mouse": "mouse",
    "controle": "remote",
    "teclado": "keyboard",
    "celular": "cell phone",
    "telefone": "cell phone",
    "microondas": "microwave",
    "forno": "oven",
    "torradeira": "toaster",
    "pia": "sink",
    "geladeira": "refrigerator",
    # household
    "livro": "book",
    "livros": "book",
    "relogio": "clock",
    "relógio": "clock",
    "tesoura": "scissors",
    "ursinho": "teddy bear",
    "secador": "hair drier",
    "escova": "toothbrush",
}


def with_coco_aliases(query_tokens: list[str]) -> list[str]:
    """Rewrite Portuguese query tokens to their English COCO class tokens.

    Tokens without an alias pass through unchanged, so English queries keep
    working. Aliased tokens are REPLACED rather than added: the caller's phrase
    matcher requires every query token to be present in the class name, so
    carrying both languages would make each alias unmatchable.
    """
    if not query_tokens:
        return query_tokens
    out: list[str] = []
    for token in query_tokens:
        english = _PT_TO_COCO.get(token)
        out.extend(tokenize(english) if english else [token])
    return out
