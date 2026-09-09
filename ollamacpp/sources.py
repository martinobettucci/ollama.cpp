"""Sources de modèles : Hugging Face et registre privé.

@spec docs/BACKLOG.md OC-050 « /api/pull », OC-061 « Source Hugging Face »,
      OC-062 « Registre privé natif »
@spec docs/ollama.cpp-architecture.md §5.10 « Arborescence de données », §8 risques R8 et R9
@spec docs/DAT.md §3.3 « Téléchargement », §7 « Sécurité »

Un `pull` résout un nom logique vers une source, télécharge les artefacts dans le magasin
adressé par contenu, puis écrit le manifest :

    nom logique → source → artefacts → cache local → manifest

**Risque R9 — secrets.** Les jetons de registre ne transitent que dans l'en-tête `Authorization`
et ne sont **jamais** journalisés : les traces d'événements ne portent que l'hôte et le chemin.

**Risque R8 — chemins.** Aucun nom de fichier distant ne devient un chemin d'écriture. Le
`BlobStore` nomme les fichiers d'après le digest qu'il calcule lui-même ; un artefact dont le
checksum annoncé ne correspond pas est détruit sans être installé.

Le format de progression est celui d'Ollama (`api.ProgressResponse` : `status`, `digest`,
`total`, `completed`), afin que `ollama pull` et la console d'`ollama-gateway` affichent une
progression réelle.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any, TYPE_CHECKING

import httpx

from .observability import EVENT_DOWNLOAD_COMPLETE, EVENT_DOWNLOAD_STARTED, emit
from .storage.blobs import BlobError
from .storage.manifests import Artifacts, Manifest, ModelSource

if TYPE_CHECKING:  # pragma: no cover - uniquement pour le typage
    from .service import Service

#: Taille des blocs de téléchargement. Compromis entre appels système et pic mémoire.
CHUNK_SIZE = 1024 * 1024

#: `hf.co/<dépôt>` et `huggingface.co/<dépôt>` sont les préfixes qu'Ollama reconnaît déjà pour
#: désigner un dépôt Hugging Face ; les accepter évite d'inventer une syntaxe concurrente.
_HF_PREFIXES = ("hf.co/", "huggingface.co/")

#: Un nom de fichier d'artefact distant doit rester un nom simple : ni séparateur, ni « .. ».
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


class PullError(Exception):
    """Échec de résolution ou de téléchargement d'un modèle."""


def _safe_filename(name: str) -> str:
    """Valide un nom de fichier annoncé par une source distante (risque R8).

    Le nom ne sert qu'à choisir le rôle de l'artefact et à l'afficher ; il ne devient jamais un
    chemin. La validation est néanmoins faite ici pour qu'aucune valeur hostile ne circule.
    """
    if not _SAFE_FILENAME.match(name) or name in (".", ".."):
        raise PullError(f"invalid artifact name from remote source: {name!r}")
    return name


async def pull_model(service: "Service", name: str) -> AsyncIterator[dict[str, Any]]:
    """Télécharge un modèle et écrit son manifest, en produisant des événements de progression."""
    ref = service.registry.resolve(name)
    emit(EVENT_DOWNLOAD_STARTED, model=ref.display_shortest())

    yield {"status": "pulling manifest"}

    source, artifacts_spec = await _resolve(service, name)

    digests: dict[str, str] = {}
    for role, spec in artifacts_spec.items():
        async for event in _fetch_artifact(service, role, spec, digests):
            yield event

    if "model" not in digests:
        raise PullError("no model artifact found for this source")

    manifest = Manifest(
        name=ref,
        artifacts=Artifacts(model=digests["model"], mmproj=digests.get("mmproj")),
        source=source,
    )
    service.registry.install(manifest)

    emit(EVENT_DOWNLOAD_COMPLETE, model=ref.display_shortest(), artifacts=len(digests))
    yield {"status": "verifying sha256 digest"}
    yield {"status": "writing manifest"}
    yield {"status": "success"}


async def _resolve(
    service: "Service", name: str
) -> tuple[ModelSource, dict[str, dict[str, Any]]]:
    """Résout un nom logique vers une source et la liste de ses artefacts.

    Deux formes sont reconnues, toutes deux des **noms de modèles Ollama valides** :

    - `hf.co/<propriétaire>/<dépôt>` ou `huggingface.co/…` — source « huggingface » ;
    - tout autre nom, si un registre privé est configuré — source « private ».

    Un chemin de fichier local n'en est délibérément pas une : ce n'est pas un nom de modèle
    valide, et Ollama ne l'accepte pas non plus sur `/api/pull`. L'installation d'un GGUF local
    passe par `POST /api/blobs/<digest>` puis `POST /api/create` avec `files` — le chemin natif
    d'Ollama, déjà implémenté (OC-053, OC-054).
    """
    lowered = name.lower()
    for prefix in _HF_PREFIXES:
        if lowered.startswith(prefix):
            return await _resolve_huggingface(service, name[len(prefix):])

    if service.config.registry_url:
        return await _resolve_private(service, name)

    raise PullError(
        f"cannot resolve model '{name}': use an 'hf.co/<owner>/<repo>' reference, configure "
        "OLLAMACPP_REGISTRY_URL for a private registry, or install a local GGUF with "
        "/api/blobs and /api/create"
    )


async def _resolve_huggingface(
    service: "Service", repo: str
) -> tuple[ModelSource, dict[str, dict[str, Any]]]:
    """Résout un dépôt Hugging Face vers ses fichiers GGUF.

    Le fichier peut être précisé après `:` (`hf.co/org/depot:Q4_K_M.gguf`) ; sinon le premier
    GGUF du dépôt qui n'est pas un projecteur est retenu, et un éventuel `mmproj` est associé.
    """
    repo_id, _, wanted = repo.partition(":")
    if not repo_id:
        raise PullError("missing Hugging Face repository name")

    endpoint = service.config.hf_endpoint.rstrip("/")
    headers = _auth_headers(service.config.hf_token)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{endpoint}/api/models/{repo_id}", headers=headers)
            if response.status_code == 404:
                raise PullError(f"Hugging Face repository not found: {repo_id}")
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise PullError(f"unable to reach Hugging Face for repository {repo_id}") from exc

    files = [
        str(item.get("rfilename"))
        for item in (payload.get("siblings") or [])
        if isinstance(item, dict) and str(item.get("rfilename", "")).endswith(".gguf")
    ]
    if not files:
        raise PullError(f"no GGUF file found in repository {repo_id}")

    projectors = [f for f in files if "mmproj" in f.lower()]
    weights = [f for f in files if f not in projectors]
    if not weights:
        raise PullError(f"repository {repo_id} contains only projector files")

    if wanted:
        # La recherche porte sur les **poids** seuls : sans cela, `:f16` pourrait retenir
        # `mmproj-…-f16.gguf` comme modèle, un projecteur n'étant pas un modèle.
        matching = [f for f in weights if f.endswith(wanted) or wanted in f]
        if not matching:
            raise PullError(f"no file matching '{wanted}' in repository {repo_id}")
        chosen = matching[0]
    else:
        chosen = sorted(weights)[0]

    artifacts: dict[str, dict[str, Any]] = {
        "model": {"kind": "url", "url": f"{endpoint}/{repo_id}/resolve/main/{chosen}",
                  "headers": headers, "name": chosen.rsplit("/", 1)[-1]}
    }
    projector = _pair_projector(chosen, projectors)
    if projector is not None:
        artifacts["mmproj"] = {
            "kind": "url",
            "url": f"{endpoint}/{repo_id}/resolve/main/{projector}",
            "headers": headers,
            "name": projector.rsplit("/", 1)[-1],
        }

    return ModelSource(type="huggingface", reference=repo_id), artifacts


def _quantization_suffix(filename: str) -> str:
    """Dernier segment d'un nom de GGUF, qui porte par convention la quantification.

    `SmolVLM-256M-Instruct-Q8_0.gguf` → `q8_0` ; `…-f16.gguf` → `f16`. C'est une convention de
    nommage, pas une garantie du format : elle sert uniquement à apparier deux fichiers d'un même
    dépôt, jamais à décider de quoi que ce soit d'autre.
    """
    base = filename.rsplit("/", 1)[-1]
    if base.lower().endswith(".gguf"):
        base = base[: -len(".gguf")]
    return base.rsplit("-", 1)[-1].lower()


def _pair_projector(chosen_weights: str, projectors: list[str]) -> str | None:
    """Choisit le projecteur qui va avec le fichier de poids retenu.

    Les dépôts de vision publient souvent un `mmproj` **par quantification**
    (`ggml-org/SmolVLM-256M-Instruct-GGUF` en est un exemple). Prendre le premier par ordre
    alphabétique livrait alors un encodeur d'image d'une autre précision que celle demandée :
    `:f16` donnait un modèle f16 et un projecteur Q8_0, sans que rien ne le signale. Le résultat
    fonctionne — `llama.cpp` accepte l'écart — ce qui rend la surprise d'autant plus silencieuse.

    À défaut de correspondance, le premier par ordre alphabétique reste retenu : un dépôt qui ne
    publie qu'un projecteur le destine à tous ses poids.
    """
    if not projectors:
        return None
    voulu = _quantization_suffix(chosen_weights)
    for projector in sorted(projectors):
        if _quantization_suffix(projector) == voulu:
            return projector
    return sorted(projectors)[0]


async def _resolve_private(
    service: "Service", name: str
) -> tuple[ModelSource, dict[str, dict[str, Any]]]:
    """Résout un modèle depuis le registre privé de `ollama.cpp`.

    Le registre expose `GET {registry}/v1/models/{nom}` renvoyant un manifest décrivant ses
    artefacts, chacun avec son URL et son checksum attendu. Le checksum est **obligatoire** :
    sans lui, rien ne distingue un artefact légitime d'un artefact substitué.
    """
    base = service.config.registry_url.rstrip("/")
    headers = _auth_headers(service.config.registry_token)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base}/v1/models/{name}", headers=headers)
            if response.status_code == 404:
                raise PullError(f"model '{name}' not found in the private registry")
            if response.status_code in (401, 403):
                raise PullError("private registry refused the credentials")
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        # Ni l'URL du registre ni le jeton n'apparaissent dans le message rendu au client.
        raise PullError(f"unable to reach the private registry for model '{name}'") from exc

    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise PullError(f"private registry returned no artifacts for model '{name}'")

    artifacts: dict[str, dict[str, Any]] = {}
    for role, spec in raw_artifacts.items():
        if not isinstance(spec, dict):
            continue
        url = spec.get("url")
        digest = spec.get("digest")
        if not isinstance(url, str) or not url:
            raise PullError(f"private registry artifact '{role}' has no URL")
        if not isinstance(digest, str) or not digest:
            raise PullError(f"private registry artifact '{role}' has no checksum")
        artifacts[str(role)] = {
            "kind": "url",
            "url": _absolute_url(base, url),
            "headers": headers,
            "digest": digest,
            "name": _safe_filename(str(spec.get("name") or f"{role}.gguf")),
        }

    return ModelSource(type="private", reference=name), artifacts


def _absolute_url(base: str, url: str) -> str:
    """Complète une URL relative fournie par le registre, sans jamais quitter ce registre."""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"{base}/{url.lstrip('/')}"


def _auth_headers(token: str) -> dict[str, str]:
    """En-têtes d'authentification. Le jeton n'est jamais journalisé (risque R9)."""
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _fetch_artifact(
    service: "Service", role: str, spec: dict[str, Any], digests: dict[str, str]
) -> AsyncIterator[dict[str, Any]]:
    """Télécharge un artefact vers le magasin et enregistre son digest."""
    expected = spec.get("digest")

    if expected and service.registry.blobs.has(expected):
        # Déduplication : un artefact déjà présent n'est pas retéléchargé. C'est ce qui rend un
        # `pull` répété quasi instantané et ce qui permet à deux modèles de partager un blob.
        digests[role] = expected
        yield {"status": f"pulling {role}", "digest": expected,
               "total": service.registry.blobs.size(expected),
               "completed": service.registry.blobs.size(expected)}
        return

    url = spec["url"]
    headers = spec.get("headers") or {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0),
                                     follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code >= 400:
                    raise PullError(f"failed to download {role}: HTTP {response.status_code}")
                total = int(response.headers.get("content-length") or 0)
                yield {"status": f"pulling {role}", "total": total, "completed": 0}

                collected: list[bytes] = []
                completed = 0
                async for chunk in response.aiter_bytes(CHUNK_SIZE):
                    collected.append(chunk)
                    completed += len(chunk)
                    yield {"status": f"pulling {role}", "total": total, "completed": completed}

        info = service.registry.blobs.ingest_chunks(collected, expected_digest=expected)
    except httpx.HTTPError as exc:
        raise PullError(f"failed to download {role}") from exc
    except BlobError as exc:
        # Un checksum non conforme est une erreur de sécurité, pas un simple aléa réseau.
        raise PullError(f"checksum verification failed for {role}") from exc

    digests[role] = info.digest
    yield {"status": f"pulling {role}", "digest": info.digest,
           "total": info.size, "completed": info.size}
