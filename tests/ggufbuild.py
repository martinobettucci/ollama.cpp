"""Fabrique de fichiers GGUF réels pour les tests.

@verifies docs/BACKLOG.md OC-023 « Métadonnées GGUF »
@verifies docs/ollama.cpp-architecture.md §5.6 « Ordre de précédence de la configuration »

Ces fichiers sont de **vrais** GGUF au sens du format binaire (`ggml/include/gguf.h`) : en-tête,
version, compteurs, paires clé/valeur typées. Ils ne contiennent simplement aucun tenseur, ce qui
suffit à tout ce que `ollama.cpp` lit sans charger le modèle.

Écrire l'encodeur ici plutôt que de figer un binaire opaque a un intérêt de vérification : le
lecteur d'OC-023 est confronté à un encodeur écrit **depuis la spécification**, pas depuis le
lecteur lui-même.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from ollamacpp.gguf import GGUF_MAGIC, GGUFType


def _string(value: str) -> bytes:
    payload = value.encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def _typed(value: Any) -> bytes:
    """Encode une valeur avec son type, en choisissant le type GGUF le plus naturel."""
    if isinstance(value, bool):
        return struct.pack("<i", GGUFType.BOOL) + struct.pack("<b", 1 if value else 0)
    if isinstance(value, int):
        # Comme les vrais GGUF : `general.parameter_count` d'un modèle de 7 milliards de
        # paramètres ne tient pas sur 32 bits et y est encodé en UINT64.
        if value > 0xFFFFFFFF:
            return struct.pack("<i", GGUFType.UINT64) + struct.pack("<Q", value)
        return struct.pack("<i", GGUFType.UINT32) + struct.pack("<I", value)
    if isinstance(value, float):
        return struct.pack("<i", GGUFType.FLOAT32) + struct.pack("<f", value)
    if isinstance(value, str):
        return struct.pack("<i", GGUFType.STRING) + _string(value)
    if isinstance(value, list):
        if not value:
            return (struct.pack("<i", GGUFType.ARRAY) + struct.pack("<I", GGUFType.STRING)
                    + struct.pack("<Q", 0))
        if all(isinstance(item, str) for item in value):
            body = b"".join(_string(item) for item in value)
            element_type = GGUFType.STRING
        elif all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            body = b"".join(struct.pack("<I", item) for item in value)
            element_type = GGUFType.UINT32
        else:
            raise TypeError("tableau GGUF hétérogène non supporté par la fabrique de test")
        return (struct.pack("<i", GGUFType.ARRAY) + struct.pack("<I", element_type)
                + struct.pack("<Q", len(value)) + body)
    raise TypeError(f"type non supporté par la fabrique de test : {type(value).__name__}")


def build_gguf(kv: dict[str, Any], *, version: int = 3, tensor_count: int = 0) -> bytes:
    """Sérialise un en-tête GGUF complet avec les paires clé/valeur données."""
    out = bytearray(GGUF_MAGIC)
    out += struct.pack("<I", version)
    out += struct.pack("<q", tensor_count)
    out += struct.pack("<q", len(kv))
    for key, value in kv.items():
        out += _string(key)
        out += _typed(value)
    return bytes(out)


#: Modèle de test minimal, représentatif d'un GGUF réel de type Qwen quantifié en Q4_K_M.
DEFAULT_KV: dict[str, Any] = {
    "general.architecture": "qwen3",
    "general.name": "Qwen3 Test",
    "general.file_type": 15,  # LLAMA_FTYPE_MOSTLY_Q4_K_M
    "general.parameter_count": 7_600_000_000,
    "qwen3.context_length": 32768,
    "qwen3.embedding_length": 4096,
    "qwen3.block_count": 36,
    "tokenizer.chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    "tokenizer.ggml.tokens": ["<s>", "</s>", "a", "b", "c"],
}


def write_gguf(path: Path, kv: dict[str, Any] | None = None, **kwargs: Any) -> Path:
    """Écrit un GGUF de test sur disque et renvoie son chemin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_gguf(DEFAULT_KV if kv is None else kv, **kwargs))
    return path


def build_mmproj() -> bytes:
    """GGUF de projecteur multimodal, reconnaissable à ses clés `clip.*`."""
    return build_gguf(
        {
            "general.architecture": "clip",
            "clip.has_vision_encoder": True,
            "clip.vision.image_size": 336,
        }
    )
