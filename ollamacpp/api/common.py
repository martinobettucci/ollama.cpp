"""Utilitaires partagés par les quatre façades.

@spec docs/BACKLOG.md OC-011 « Schéma d'erreur compatible Ollama »
@spec docs/ollama.cpp-architecture.md §3.1 « Sonde de compatibilité », §8 risque R1
@spec docs/DAT.md §5.1 « Interfaces exposées »

Ces fonctions vivent ici — et non dans l'une des façades — pour qu'aucune façade n'ait à importer
une autre façade. La règle d'architecture est stricte : les façades convergent vers la couche
canonique, jamais l'une vers l'autre (mission §22).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request

from ..errors import BadRequest, Forbidden
from ..service import Service


def get_service(request: Request) -> Service:
    """Service partagé par toutes les façades : un seul registre, un seul runtime."""
    return request.app.state.service


async def read_body(request: Request) -> dict[str, Any]:
    """Lit un corps JSON en objet, à la manière d'Ollama.

    Un corps vide devient `{}` : c'est exactement ce qu'envoie la sonde de compatibilité
    d'`ollama-gateway` (`app/servers.py::_probe_endpoint`), et le traiter comme une erreur de
    parsage la ferait échouer. Le traitement produit alors une erreur métier — « model '' not
    found » — qui reste reconnue comme « endpoint servi » parce qu'elle contient le mot `model`
    (risque R1).
    """
    raw = await request.body()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BadRequest("invalid JSON in request body") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise BadRequest("request body must be a JSON object")
    return payload


def require_management(service: Service) -> None:
    """Applique le verrou du plan de contrôle, **côté serveur** (DAT §6).

    `ollama-gateway` refuse déjà ces chemins aux clés clientes, mais une règle appliquée
    uniquement en amont n'est pas une règle : `ollama.cpp` doit rester sûr exposé seul.
    """
    if not service.config.management_enabled:
        raise Forbidden("model management is disabled on this server")


def reject_unsupported(resident, canonical, *, requested: str | None = None) -> None:
    """Refuse une requête que le modèle chargé ne peut pas honorer, avec le message d'Ollama.

    @spec docs/BACKLOG.md OC-032 « Détection de capacités observables », OC-044, OC-071, OC-074,
          OC-075
    @spec docs/ollama.cpp-architecture.md §5.7 « Capacités » ; mission §13

    Les capacités viennent de `/props` du modèle **réellement en mémoire**, pas du manifest ni
    d'une heuristique sur le chat template : c'est la seule source qui décrit ce que le runtime
    sait faire à cet instant (OC-032).

    Cette fonction est partagée par les quatre façades, et cette mutualisation est le correctif
    d'un défaut trouvé sur un vrai modèle de vision : le contrôle n'existait que sur les façades
    Ollama et OpenAI. Sur Responses et Anthropic, l'image atteignait `llama-server`, qui la
    refusait ; le client recevait un `502` porteur d'un message d'amont conseillant de fournir un
    projecteur. Un `502` annonce une panne du serveur là où la requête est simplement invalide, et
    ce conseil s'adresse à l'exploitant, pas au client. Un garde-fou par façade est un garde-fou
    qu'on oublie : il n'y en a plus qu'un.

    `requested` permet de nommer le modèle tel que le client l'a demandé, plutôt que sous son nom
    résolu — c'est ce que fait Ollama dans ses messages. À défaut, le nom du modèle résident.
    """
    nom = requested or resident.name
    capabilities = resident.capabilities
    if canonical.thinking.is_requested and "thinking" not in capabilities:
        raise BadRequest(f'"{nom}" does not support thinking')
    if canonical.tools and "tools" not in capabilities:
        raise BadRequest(f'"{nom}" does not support tools')
    if any(getattr(message, "images", ()) for message in canonical.messages) and (
        "vision" not in capabilities
    ):
        raise BadRequest(f'"{nom}" does not support vision')
