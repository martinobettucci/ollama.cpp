"""Lecture des métadonnées GGUF.

@spec docs/BACKLOG.md OC-023 « Métadonnées GGUF »
@spec docs/ollama.cpp-architecture.md §5.6 « Ordre de précédence de la configuration »
@spec docs/DAT.md §4.3 « Précédence de configuration »

Le GGUF est la **source de vérité privilégiée** : architecture, tokenizer, chat template et
contexte natif y sont déjà décrits, et la mission interdit de les dupliquer dans le manifest
(§11 de la mission). Ce module lit l'en-tête sans charger le modèle, ce qui permet de servir
`/api/tags` et `/api/show` sur des modèles non résidents.

Format lu, tel que documenté dans `ggml/include/gguf.h` (révision auditée `39be55c`) :

1. magie `GGUF` (4 octets) ;
2. version (`uint32`) ;
3. nombre de tenseurs (`int64`) ;
4. nombre de paires clé/valeur (`int64`) ;
5. pour chaque paire : clé (chaîne), type (`int32`), puis la valeur — un tableau étant précédé
   du type de ses éléments et de leur nombre (`uint64`).

Les chaînes sont sérialisées en `uint64` de longueur suivi des octets, sans terminateur nul ;
les booléens tiennent sur un `int8` ; les énumérations sur un `int32`.

Après les paires clé/valeur vient la table des tenseurs — nom, forme, type, décalage — puis le blob
de poids. Seules les descriptions sont lues : le blob n'est jamais chargé en mémoire. Le compte de
paramètres en est déduit et écrit dans `general.parameter_count`, comme le fait Ollama.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

#: Au-delà, on considère le fichier corrompu plutôt que d'allouer sans borne (défense contre un
#: GGUF hostile qui annoncerait des tailles absurdes).
_MAX_KV_COUNT = 1_000_000
_MAX_STRING_LEN = 64 * 1024 * 1024
_MAX_ARRAY_LEN = 100_000_000

#: `MaxTensorDims` de `fs/gguf/gguf.go` (Ollama) et de `ggml` : un tenseur a au plus 4 dimensions.
_MAX_TENSOR_DIMS = 4


class GGUFError(ValueError):
    """Fichier GGUF illisible ou malformé."""


class GGUFType(IntEnum):
    """`enum gguf_type` de `ggml/include/gguf.h`."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FORMATS: dict[int, tuple[str, int]] = {
    GGUFType.UINT8: ("<B", 1),
    GGUFType.INT8: ("<b", 1),
    GGUFType.UINT16: ("<H", 2),
    GGUFType.INT16: ("<h", 2),
    GGUFType.UINT32: ("<I", 4),
    GGUFType.INT32: ("<i", 4),
    GGUFType.FLOAT32: ("<f", 4),
    GGUFType.BOOL: ("<b", 1),
    GGUFType.UINT64: ("<Q", 8),
    GGUFType.INT64: ("<q", 8),
    GGUFType.FLOAT64: ("<d", 8),
}

#: `enum llama_ftype` (`include/llama.h`) → nom court, aligné sur `fs/ggml/type.go` d'Ollama pour
#: que `details.quantization_level` affiche exactement ce qu'affiche Ollama.
FILE_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0",
}


def file_type_name(value: int | None) -> str:
    """Nom court d'un `general.file_type`. `unknown` plutôt qu'une erreur : un ftype inconnu
    d'une version future de `llama.cpp` ne doit pas empêcher de lister le modèle."""
    if value is None:
        return "unknown"
    return FILE_TYPE_NAMES.get(int(value), "unknown")


def human_number(count: int) -> str:
    """Formate un nombre de paramètres comme `format.HumanNumber` d'Ollama (`format/format.go`).

    Reproduit les seuils et le nombre de décimales exacts : `7.6B`, `109M`, `560K`.
    """
    billion, million, thousand = 1_000_000_000, 1_000_000, 1_000
    if count >= billion:
        number = count / billion
        return f"{number:.0f}B" if number == int(number) else f"{number:.1f}B"
    if count >= million:
        number = count / million
        return f"{number:.0f}M" if number == int(number) else f"{number:.2f}M"
    if count >= thousand:
        return f"{count / thousand:.0f}K"
    return str(count)


# --- Lecture -------------------------------------------------------------------------------------


def _read_exactly(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise GGUFError("fin de fichier prématurée")
    return data


def _read_scalar(stream: BinaryIO, value_type: int) -> Any:
    spec = _SCALAR_FORMATS.get(value_type)
    if spec is None:
        raise GGUFError(f"type GGUF inconnu : {value_type}")
    fmt, size = spec
    value = struct.unpack(fmt, _read_exactly(stream, size))[0]
    return bool(value) if value_type == GGUFType.BOOL else value


def _read_string(stream: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read_exactly(stream, 8))
    if length > _MAX_STRING_LEN:
        raise GGUFError("chaîne GGUF démesurée")
    # `errors="replace"` : une chaîne mal encodée ne doit pas rendre tout le modèle illisible.
    return _read_exactly(stream, length).decode("utf-8", errors="replace")


def _read_value(stream: BinaryIO, value_type: int) -> Any:
    if value_type == GGUFType.STRING:
        return _read_string(stream)
    if value_type == GGUFType.ARRAY:
        (element_type,) = struct.unpack("<I", _read_exactly(stream, 4))
        (count,) = struct.unpack("<Q", _read_exactly(stream, 8))
        if count > _MAX_ARRAY_LEN:
            raise GGUFError("tableau GGUF démesuré")
        return [_read_value(stream, element_type) for _ in range(count)]
    return _read_scalar(stream, value_type)


@dataclass(frozen=True, slots=True)
class GGUFMetadata:
    """Métadonnées d'en-tête d'un fichier GGUF."""

    version: int
    tensor_count: int
    kv: dict[str, Any] = field(default_factory=dict)
    file_size: int = 0

    # --- Accès normalisé ------------------------------------------------------------------------

    @property
    def architecture(self) -> str:
        value = self.kv.get("general.architecture")
        return str(value) if value is not None else ""

    @property
    def name(self) -> str:
        value = self.kv.get("general.name")
        return str(value) if value is not None else ""

    @property
    def quantization_level(self) -> str:
        return file_type_name(self.kv.get("general.file_type"))

    @property
    def parameter_count(self) -> int:
        """Nombre de paramètres, **calculé** depuis la table des tenseurs.

        `read_metadata` écrit `general.parameter_count` dans `kv` après avoir additionné les
        éléments de chaque tenseur, exactement comme Ollama (`fs/ggml/gguf.go` l. 239-251). Lire
        la clé revient donc à lire le calcul, y compris quand le fichier ne la portait pas —
        c'est le cas de beaucoup de GGUF publiés, dont ceux de Qwen.
        """
        value = self.kv.get("general.parameter_count")
        return int(value) if isinstance(value, (int, float)) else 0

    @property
    def parameter_size(self) -> str:
        count = self.parameter_count
        return human_number(count) if count else ""

    @property
    def context_length(self) -> int:
        """Contexte natif, sous la clé `<architecture>.context_length`.

        La clé est préfixée par l'architecture (`llama.context_length`,
        `qwen3.context_length`…) : il faut donc connaître l'architecture pour la lire.
        """
        return self._arch_int("context_length")

    @property
    def embedding_length(self) -> int:
        return self._arch_int("embedding_length")

    @property
    def block_count(self) -> int:
        return self._arch_int("block_count")

    @property
    def chat_template(self) -> str:
        value = self.kv.get("tokenizer.chat_template")
        return str(value) if value is not None else ""

    @property
    def is_multimodal_projector(self) -> bool:
        """Vrai pour un fichier `mmproj`, reconnaissable à ses clés `clip.*`."""
        return any(key.startswith("clip.") for key in self.kv)

    def _arch_int(self, suffix: str) -> int:
        architecture = self.architecture
        if not architecture:
            return 0
        value = self.kv.get(f"{architecture}.{suffix}")
        return int(value) if isinstance(value, (int, float)) else 0

    def public_model_info(self) -> dict[str, Any]:
        """Métadonnées exposables par `/api/show.model_info`.

        Les tableaux volumineux du tokenizer (`tokenizer.ggml.tokens`, `merges`, `scores`…) sont
        remplacés par leur longueur : Ollama fait de même, et transporter un vocabulaire de
        150 000 entrées dans chaque réponse `/api/show` serait absurde.
        """
        out: dict[str, Any] = {}
        for key, value in self.kv.items():
            if isinstance(value, list):
                out[f"{key}.length"] = len(value)
            else:
                out[key] = value
        return out


def _read_tensor_table(stream: BinaryIO, tensor_count: int) -> int:
    """Parcourt la table des tenseurs et renvoie le nombre total de paramètres.

    Seules les **descriptions** sont lues — nom, forme, type, décalage — soit quelques dizaines
    d'octets par tenseur ; le blob de poids qui suit n'est jamais touché. Le compte de paramètres
    est la somme des produits des dimensions, comme `Tensor.elements()` d'Ollama
    (`fs/ggml/ggml.go` l. 523-532).

    Une table tronquée fait échouer la lecture plutôt que de renvoyer un compte partiel : un
    compte faux serait affiché comme un fait par `ollama show`.
    """
    total = 0
    for _ in range(tensor_count):
        _read_string(stream)  # nom du tenseur, sans usage ici
        (dimensions,) = struct.unpack("<I", _read_exactly(stream, 4))
        if dimensions > _MAX_TENSOR_DIMS:
            raise GGUFError(f"tenseur à {dimensions} dimensions : maximum {_MAX_TENSOR_DIMS}")
        elements = 1
        for _ in range(dimensions):
            (extent,) = struct.unpack("<Q", _read_exactly(stream, 8))
            elements *= extent
        _read_exactly(stream, 4)  # type ggml
        _read_exactly(stream, 8)  # décalage dans le blob
        total += elements
    return total


def read_metadata(path: Path | str) -> GGUFMetadata:
    """Lit l'en-tête GGUF d'un fichier. Ne charge jamais le blob de tenseurs."""
    file_path = Path(path)
    with file_path.open("rb") as stream:
        magic = _read_exactly(stream, 4)
        if magic != GGUF_MAGIC:
            raise GGUFError(f"magie GGUF absente dans {file_path.name}")

        (version,) = struct.unpack("<I", _read_exactly(stream, 4))
        (tensor_count,) = struct.unpack("<q", _read_exactly(stream, 8))
        (kv_count,) = struct.unpack("<q", _read_exactly(stream, 8))

        if kv_count < 0 or kv_count > _MAX_KV_COUNT:
            raise GGUFError(f"nombre de paires clé/valeur invalide : {kv_count}")
        if tensor_count < 0:
            raise GGUFError(f"nombre de tenseurs invalide : {tensor_count}")

        kv: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(stream)
            (value_type,) = struct.unpack("<i", _read_exactly(stream, 4))
            kv[key] = _read_value(stream, value_type)

        parameters = _read_tensor_table(stream, tensor_count)
        if tensor_count:
            kv["general.parameter_count"] = parameters

    return GGUFMetadata(
        version=version,
        tensor_count=tensor_count,
        kv=kv,
        file_size=file_path.stat().st_size,
    )
