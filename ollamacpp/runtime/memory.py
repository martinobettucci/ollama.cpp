"""Estimation et mesure de la mémoire.

@spec docs/BACKLOG.md OC-034 « ModelScheduler »
@spec docs/ollama.cpp-architecture.md §5.8 « Scheduler », §8 risques R6 et R11
@spec docs/DAT.md §1 « Composants »

**Risque R6 — estimation imprécise.** Sous-estimer coûte un OOM, qui peut faire tomber l'hôte
entier ; surestimer coûte de la place inutilisée. L'estimation penche donc systématiquement du
côté prudent, et une marge de sécurité configurable s'y ajoute.

**Risque R11 — pas de GPU dans l'environnement de développement de référence.** La mesure de la
mémoire disponible est isolée derrière `MemoryProbe`, ce qui permet de tester les décisions du
scheduler par injection, sans dépendre du matériel de la machine de test.

L'estimation du cache KV suit la structure réelle d'un modèle transformeur :

    octets_kv = 2 (K et V) × contexte × couches × dimension_kv × octets_par_élément

`dimension_kv` tient compte de la *grouped-query attention* quand le GGUF publie
`attention.head_count` et `attention.head_count_kv` : sans cela, l'estimation serait fausse d'un
facteur 8 sur les modèles récents, ce qui interdirait de charger des modèles qui tiennent
largement en mémoire.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from ..gguf import GGUFMetadata

#: Octets par élément du cache KV selon le type. Valeurs volontairement arrondies **vers le
#: haut** : une sous-estimation coûte un OOM, une surestimation coûte de la place (risque R6).
CACHE_TYPE_BYTES: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.125,
    "q5_0": 0.75,
    "q5_1": 0.8125,
    "q4_0": 0.625,
    "q4_1": 0.6875,
    "iq4_nl": 0.625,
}

DEFAULT_CACHE_TYPE = "f16"

#: Surcoût fixe d'un processus `llama-server` : contextes de calcul, tampons de sortie, runtime.
#: Mesure grossière assumée, documentée plutôt que cachée dans une constante magique.
PROCESS_OVERHEAD_BYTES = 256 * 1024 * 1024


def cache_type_bytes(name: str | None) -> float:
    """Octets par élément d'un type de cache. Le défaut prudent est `f16`."""
    if not name:
        return CACHE_TYPE_BYTES[DEFAULT_CACHE_TYPE]
    return CACHE_TYPE_BYTES.get(name.lower(), CACHE_TYPE_BYTES[DEFAULT_CACHE_TYPE])


def estimate_kv_bytes(
    metadata: GGUFMetadata | None,
    *,
    context: int,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    parallel: int = 1,
) -> int:
    """Estime la taille du cache KV.

    Renvoie `0` si le GGUF ne publie pas les dimensions nécessaires : mieux vaut une estimation
    ouvertement incomplète, que l'appelant peut détecter, qu'un chiffre inventé.
    """
    if metadata is None or context <= 0:
        return 0

    layers = metadata.block_count
    embedding = metadata.embedding_length
    if layers <= 0 or embedding <= 0:
        return 0

    # Grouped-query attention : la dimension du cache suit le nombre de têtes *KV*, pas le nombre
    # de têtes d'attention. Sans cette correction l'estimation serait fausse d'un facteur égal au
    # ratio de groupement (souvent 4 à 8 sur les modèles récents).
    architecture = metadata.architecture
    heads = metadata.kv.get(f"{architecture}.attention.head_count")
    heads_kv = metadata.kv.get(f"{architecture}.attention.head_count_kv")
    kv_dimension = embedding
    if isinstance(heads, int) and isinstance(heads_kv, int) and heads > 0 and heads_kv > 0:
        kv_dimension = int(embedding * heads_kv / heads)

    per_element = cache_type_bytes(cache_type_k) + cache_type_bytes(cache_type_v)
    slots = max(1, parallel)
    return int(context * layers * kv_dimension * per_element * slots)


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Estimation de l'empreinte d'un modèle résident."""

    weights_bytes: int
    kv_bytes: int
    overhead_bytes: int = PROCESS_OVERHEAD_BYTES

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_bytes + self.overhead_bytes

    def as_fields(self) -> dict[str, int]:
        """Champs journalisables, pour rendre une décision d'admission explicable (OC-035)."""
        return {
            "weights_bytes": self.weights_bytes,
            "kv_bytes": self.kv_bytes,
            "overhead_bytes": self.overhead_bytes,
            "total_bytes": self.total_bytes,
        }


def estimate_model_memory(
    *,
    artifacts_bytes: int,
    metadata: GGUFMetadata | None,
    context: int,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    parallel: int = 1,
) -> MemoryEstimate:
    """Estime l'empreinte totale d'un modèle : poids sur disque + cache KV + surcoût process."""
    return MemoryEstimate(
        weights_bytes=max(0, artifacts_bytes),
        kv_bytes=estimate_kv_bytes(
            metadata,
            context=context,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            parallel=parallel,
        ),
    )


class MemoryProbe(Protocol):
    """Source de la mémoire disponible.

    Abstraite pour deux raisons : rendre les décisions du scheduler testables par injection, et
    permettre de brancher plus tard une sonde VRAM sans toucher au scheduler (risque R11).
    """

    def total_bytes(self) -> int: ...


class HostMemoryProbe:
    """Mémoire de l'hôte, lue depuis le système d'exploitation.

    Compte la mémoire **totale** et non la mémoire libre : la mémoire libre varie à chaque
    instant, ce qui rendrait les décisions d'admission non reproductibles et donc inexplicables.
    Le budget effectif est ensuite réduit par la marge de sécurité configurée.
    """

    def total_bytes(self) -> int:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return 0


class FixedMemoryProbe:
    """Budget mémoire fixe, imposé par la configuration ou par un test."""

    def __init__(self, total: int) -> None:
        self._total = total

    def total_bytes(self) -> int:
        return self._total


def resolve_memory_budget(
    probe: MemoryProbe, *, configured_limit: int, safety_margin: float
) -> int:
    """Budget mémoire effectif du scheduler.

    Une limite explicite l'emporte toujours sur la mesure : c'est le moyen pour l'exploitant de
    cantonner `ollama.cpp` sur une machine partagée. Dans les deux cas la marge de sécurité est
    appliquée.
    """
    total = configured_limit if configured_limit > 0 else probe.total_bytes()
    if total <= 0:
        return 0
    return int(total * (1.0 - safety_margin))
