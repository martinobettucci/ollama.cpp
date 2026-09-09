"""Ordonnancement de la résidence mémoire des modèles.

@spec docs/BACKLOG.md OC-034 « ModelScheduler », OC-035 « Observabilité des décisions »
@spec docs/ollama.cpp-architecture.md §5.8 « Scheduler », §8 risque R6
@spec docs/DAT.md §3.2 « Chargement d'un modèle »

Le scheduler **décide**, il n'exécute pas : il reçoit l'état des résidents et renvoie un plan.
Cette séparation le rend testable sans processus ni mémoire réels, et évite une dépendance
circulaire avec le gestionnaire de cycle de vie qui, lui, exécute le plan.

Politique, telle que définie par la mission (§18) :

    modèle READY                      → réutiliser
    modèle déchargé + mémoire libre   → charger
    pression mémoire                  → chercher un candidat IDLE à évincer
    keep_alive expiré                 → candidat privilégié
    sinon                             → LRU pondéré par la priorité

Deux règles absolues, appliquées avant toute autre considération :

- **un modèle BUSY n'est jamais évincé** — l'évincer tuerait une requête en cours ;
- **le modèle demandé n'est jamais évincé pour se faire de la place à lui-même.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..observability import EVENT_EVICTION, emit


@dataclass(frozen=True, slots=True)
class ResidentInfo:
    """Vue du scheduler sur un modèle résident."""

    name: str
    estimated_bytes: int
    active_requests: int = 0
    idle_seconds: float = 0.0
    priority: int = 0
    keep_alive_expired: bool = False

    #: Vrai pour un modèle qu'une raison structurelle rend inévinçable même sans requête active :
    #: un modèle en cours de chargement a déjà réservé sa mémoire et l'évincer laisserait un
    #: processus à moitié démarré.
    pinned: bool = False

    @property
    def is_busy(self) -> bool:
        return self.active_requests > 0 or self.pinned


@dataclass(frozen=True, slots=True)
class EvictionDecision:
    """Décision d'éviction, avec la raison qui la rend explicable (mission §30)."""

    model: str
    reason: str
    idle_seconds: float
    priority: int

    def as_log_fields(self) -> dict[str, object]:
        return {
            "model": self.model,
            "action": "evict",
            "reason": self.reason,
            "idle_seconds": self.idle_seconds,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class AdmissionPlan:
    """Plan d'admission d'un modèle : ce qu'il faut évincer, ou pourquoi c'est impossible."""

    admitted: bool
    evictions: tuple[EvictionDecision, ...] = ()
    reason: str = ""
    freed_bytes: int = 0
    required_bytes: int = 0
    available_bytes: int = 0

    @property
    def needs_eviction(self) -> bool:
        return bool(self.evictions)


REASON_ALREADY_RESIDENT = "already_resident"
REASON_ENOUGH_MEMORY = "enough_memory"
REASON_MEMORY_PRESSURE = "memory_pressure"
REASON_SLOT_PRESSURE = "max_loaded_models"
REASON_KEEP_ALIVE_EXPIRED = "keep_alive_expired"
REASON_INSUFFICIENT_MEMORY = "insufficient_memory"


class ModelScheduler:
    """Décide de la résidence des modèles."""

    def __init__(self, *, memory_budget_bytes: int, max_loaded_models: int) -> None:
        self._budget = memory_budget_bytes
        self._max_loaded = max_loaded_models

    @property
    def memory_budget_bytes(self) -> int:
        return self._budget

    @property
    def max_loaded_models(self) -> int:
        return self._max_loaded

    # --- Décision --------------------------------------------------------------------------------

    def plan(
        self, *, name: str, required_bytes: int, residents: list[ResidentInfo]
    ) -> AdmissionPlan:
        """Calcule le plan d'admission d'un modèle."""
        if any(resident.name == name for resident in residents):
            return AdmissionPlan(admitted=True, reason=REASON_ALREADY_RESIDENT)

        used = sum(resident.estimated_bytes for resident in residents)
        available = max(0, self._budget - used) if self._budget > 0 else math.inf

        # Un budget nul signifie « mesure indisponible » : on ne bloque pas le service sur une
        # sonde mémoire défaillante, seule la limite de nombre de modèles s'applique alors.
        memory_ok = self._budget <= 0 or required_bytes <= available
        slots_ok = len(residents) < self._max_loaded

        if memory_ok and slots_ok:
            return AdmissionPlan(
                admitted=True,
                reason=REASON_ENOUGH_MEMORY,
                required_bytes=required_bytes,
                available_bytes=int(available) if available != math.inf else 0,
            )

        candidates = self._eviction_candidates(name, residents)
        evictions: list[EvictionDecision] = []
        freed = 0

        for candidate in candidates:
            if self._is_satisfied(
                residents_count=len(residents) - len(evictions),
                available=available + freed,
                required=required_bytes,
            ):
                break
            reason = (
                REASON_KEEP_ALIVE_EXPIRED
                if candidate.keep_alive_expired
                else (REASON_MEMORY_PRESSURE if not memory_ok else REASON_SLOT_PRESSURE)
            )
            evictions.append(
                EvictionDecision(
                    model=candidate.name,
                    reason=reason,
                    idle_seconds=candidate.idle_seconds,
                    priority=candidate.priority,
                )
            )
            freed += candidate.estimated_bytes

        satisfied = self._is_satisfied(
            residents_count=len(residents) - len(evictions),
            available=available + freed,
            required=required_bytes,
        )

        if not satisfied:
            # Distinguer les deux impossibilités : « aucun candidat évinçable » (tout est BUSY)
            # et « le modèle ne tient pas, même seul ». Le message d'erreur en dépend.
            return AdmissionPlan(
                admitted=False,
                evictions=(),
                reason=REASON_INSUFFICIENT_MEMORY,
                required_bytes=required_bytes,
                available_bytes=int(available) if available != math.inf else 0,
            )

        return AdmissionPlan(
            admitted=True,
            evictions=tuple(evictions),
            reason=evictions[0].reason if evictions else REASON_ENOUGH_MEMORY,
            freed_bytes=freed,
            required_bytes=required_bytes,
            available_bytes=int(available) if available != math.inf else 0,
        )

    def _is_satisfied(self, *, residents_count: int, available: float, required: int) -> bool:
        memory_ok = self._budget <= 0 or required <= available
        return memory_ok and residents_count < self._max_loaded

    def _eviction_candidates(
        self, requested: str, residents: list[ResidentInfo]
    ) -> list[ResidentInfo]:
        """Classe les résidents évinçables, du meilleur candidat au moins bon.

        Ordre de tri :

        1. `keep_alive` expiré d'abord — ces modèles auraient dû partir de toute façon ;
        2. priorité croissante — un modèle prioritaire résiste plus longtemps ;
        3. inactivité décroissante — le plus anciennement utilisé part en premier (LRU).

        Les modèles BUSY et le modèle demandé lui-même sont exclus, jamais classés.
        """
        eligible = [
            resident
            for resident in residents
            if not resident.is_busy and resident.name != requested
        ]
        return sorted(
            eligible,
            key=lambda resident: (
                not resident.keep_alive_expired,
                resident.priority,
                -resident.idle_seconds,
            ),
        )

    # --- Entretien -------------------------------------------------------------------------------

    def expired(self, residents: list[ResidentInfo]) -> list[EvictionDecision]:
        """Modèles dont le `keep_alive` a expiré et qui ne servent aucune requête.

        Appelée périodiquement : c'est ce qui rend `keep_alive` effectif même sans nouvelle
        requête pour déclencher une admission.
        """
        return [
            EvictionDecision(
                model=resident.name,
                reason=REASON_KEEP_ALIVE_EXPIRED,
                idle_seconds=resident.idle_seconds,
                priority=resident.priority,
            )
            for resident in residents
            if resident.keep_alive_expired and not resident.is_busy
        ]

    @staticmethod
    def log(decision: EvictionDecision) -> str:
        """Journalise une éviction au format explicable exigé par la mission §30."""
        return emit(EVENT_EVICTION, **decision.as_log_fields())
