#!/usr/bin/env python3
"""Seed de démonstration : installe un modèle utilisable dans un `ollama.cpp` fraîchement démarré.

@spec docs/BACKLOG.md OC-092 « Seed de démonstration reproductible »
@spec docs/DAT.md §13 « Données de développement »
@spec README.md « Commandes principales »

Le seed passe **par les mêmes API HTTP que les clients** (`/api/pull`, `/api/tags`, `/api/show`,
`/api/chat`) plutôt que d'écrire directement dans le magasin. C'est la règle §8 de CLAUDE.md :
les données de démonstration doivent être créées par le véritable flux applicatif, sinon elles ne
démontrent rien — un manifest écrit à la main prouverait seulement que le disque fonctionne.

Le modèle par défaut est volontairement minuscule (~400 Mio, Apache-2.0) pour que l'environnement
de développement reste utilisable sans GPU et sans compte.

Usage :

    python scripts/seed.py [--host http://localhost:11434] [--model hf.co/<dépôt>] [--verify]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

#: Qwen2.5 0.5B Instruct quantifié en Q4_K_M : le plus petit modèle de chat réellement utilisable,
#: sous licence Apache-2.0, sans acceptation de conditions ni compte Hugging Face.
DEFAULT_MODEL = "hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m.gguf"

#: Nom court sous lequel le modèle est ensuite exposé, pour que les exemples restent lisibles.
DEFAULT_ALIAS = "demo:latest"


def request(host: str, method: str, path: str, payload: dict | None = None,
            timeout: float = 1800.0) -> tuple[int, str]:
    """Appelle l'API du service. Renvoie (code HTTP, corps)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{host.rstrip('/')}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def wait_for_service(host: str, timeout: float = 60.0) -> bool:
    """Attend que le service réponde. `/api/version` est public et ne charge aucun modèle."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _body = request(host, "GET", "/api/version", timeout=3.0)
            if status == 200:
                return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(1.0)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="référence à télécharger (défaut : %(default)s)")
    parser.add_argument("--alias", default=DEFAULT_ALIAS,
                        help="nom court donné au modèle (défaut : %(default)s)")
    parser.add_argument("--verify", action="store_true",
                        help="exécute une inférence de contrôle après installation")
    args = parser.parse_args()

    print(f"→ attente du service sur {args.host}")
    if not wait_for_service(args.host):
        print("échec : le service ne répond pas", file=sys.stderr)
        return 1

    status, body = request(args.host, "GET", "/api/tags")
    installed = {m.get("name") for m in json.loads(body).get("models", [])}
    if args.alias in installed:
        print(f"→ {args.alias} déjà installé, rien à faire")
        return 0

    print(f"→ téléchargement de {args.model} (peut prendre plusieurs minutes)")
    status, body = request(args.host, "POST", "/api/pull",
                           {"model": args.model, "stream": False})
    if status != 200:
        print(f"échec du téléchargement (HTTP {status}) : {body}", file=sys.stderr)
        return 1

    # `copy` donne au modèle un nom court, par la même API que celle d'`ollama cp`.
    status, body = request(args.host, "POST", "/api/copy",
                           {"source": args.model, "destination": args.alias})
    if status != 200:
        print(f"échec de la copie (HTTP {status}) : {body}", file=sys.stderr)
        return 1

    status, body = request(args.host, "POST", "/api/show", {"model": args.alias})
    if status != 200:
        print(f"échec de la lecture du modèle (HTTP {status}) : {body}", file=sys.stderr)
        return 1
    details = json.loads(body)
    print(f"→ installé : {args.alias}")
    print(f"   famille       : {details['details'].get('family')}")
    print(f"   quantisation  : {details['details'].get('quantization_level')}")
    print(f"   capacités     : {', '.join(details.get('capabilities', []))}")

    if args.verify:
        print("→ inférence de contrôle")
        status, body = request(args.host, "POST", "/api/chat", {
            "model": args.alias, "stream": False,
            "messages": [{"role": "user", "content": "Réponds en un mot : bonjour"}],
        })
        if status != 200:
            print(f"échec de l'inférence (HTTP {status}) : {body}", file=sys.stderr)
            return 1
        reponse = json.loads(body)
        print(f"   réponse : {reponse['message']['content'][:120]}")
        print(f"   tokens  : {reponse.get('eval_count')} générés")

    print("→ seed terminé")
    return 0


if __name__ == "__main__":
    sys.exit(main())
