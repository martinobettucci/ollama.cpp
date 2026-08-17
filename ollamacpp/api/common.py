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
