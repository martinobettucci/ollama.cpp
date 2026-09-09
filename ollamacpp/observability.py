"""Journalisation structurée et explicable.

@spec docs/BACKLOG.md OC-035 « Observabilité des décisions »
@spec docs/ollama.cpp-architecture.md §5.8 « Scheduler », §8 risque R9 « Fuite de secrets »
@spec docs/DAT.md §7 « Sécurité »

La mission (§30) demande que les décisions du scheduler soient **explicables**, sur un format du
type :

    model=qwen3.6 action=evict reason=memory_pressure idle_seconds=731 priority=50

Ce module produit exactement cette forme : des paires `clé=valeur` stables, greppables, et
ordonnées de façon déterministe pour que deux exécutions comparables produisent des lignes
comparables.

**Risque R9.** Aucun jeton, en-tête d'authentification ni URL signée ne doit atteindre les
journaux. `emit()` masque toute valeur dont la clé ressemble à un secret : c'est un filet de
sécurité, pas une dispense de ne pas les passer.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER_NAME = "ollamacpp"

#: Fragments de noms de clés considérés comme sensibles, quelle que soit leur casse.
_SECRET_HINTS = ("token", "authorization", "api_key", "apikey", "secret", "password", "credential")

_REDACTED = "<redacted>"


def _is_secret(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    text = str(value)
    # Les espaces casseraient le découpage `clé=valeur` d'un `grep` ou d'un collecteur de logs.
    return f'"{text}"' if " " in text else text


def format_fields(fields: dict[str, Any]) -> str:
    """Sérialise des champs en `clé=valeur`, secrets masqués, ordre d'insertion préservé."""
    return " ".join(
        f"{key}={_REDACTED if _is_secret(key) else _format_value(value)}"
        for key, value in fields.items()
    )


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def emit(event: str, level: int = logging.INFO, **fields: Any) -> str:
    """Journalise un événement structuré et renvoie la ligne produite.

    Le retour permet aux tests de vérifier le contenu exact d'une décision sans dépendre de la
    configuration de journalisation de l'hôte.
    """
    line = f"event={event}"
    if fields:
        line += " " + format_fields(fields)
    get_logger().log(level, line)
    return line


def configure(level: str = "INFO") -> None:
    """Configure la journalisation du service. Idempotent."""
    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)


# --- Événements du cycle de vie et du scheduler ---------------------------------------------------
#
# Liste tenue à jour avec le §30 de la mission. Les noms sont **stables** : ils font partie du
# contrat d'exploitation au même titre qu'une API.

EVENT_MODEL_RESOLVED = "model_resolved"
EVENT_DOWNLOAD_STARTED = "download_started"
EVENT_DOWNLOAD_COMPLETE = "download_complete"
EVENT_LOAD_REQUESTED = "load_requested"
EVENT_LOAD_STARTED = "load_started"
EVENT_LOAD_COMPLETE = "load_complete"
EVENT_REQUEST_ASSIGNED = "request_assigned"
EVENT_MODEL_IDLE = "model_idle"
EVENT_KEEP_ALIVE_EXPIRED = "keep_alive_expired"
EVENT_EVICTION = "eviction"
EVENT_UNLOAD = "unload"
EVENT_FAILURE = "failure"
