"""Façade Ollama native (`/api/*`).

@spec docs/BACKLOG.md OC-040 à OC-047, OC-050 à OC-055
@spec docs/ollama.cpp-architecture.md §2.1 « Routes », §2.2 « Conventions comportementales »,
      §3.1 « Ce que la passerelle attend de l'amont », §3.2 (matrice), §5.9 « Plan de contrôle »,
      §8 risques R1 et R2
@spec docs/DAT.md §5.1 « Interfaces exposées », §6 « Authentification et autorisation »

C'est la façade prioritaire : c'est elle qui permet à un logiciel configuré pour Ollama de parler
à `ollama.cpp` sans modification.

Trois principes de mise en œuvre.

**Validation manuelle, réponses Ollama.** Les corps sont reçus en `dict` brut et validés à la
main, comme le fait Ollama. Cela produit `400 {"error": ...}` au lieu du `422 {"detail": ...}` de
FastAPI, laisse passer les champs inconnus, et surtout garantit qu'aucun endpoint servi ne
ressemble à un endpoint absent pour la sonde d'`ollama-gateway` (risque R1).

**`options.num_ctx` pilote le runtime.** La passerelle réécrit le corps pour plafonner le
contexte d'une clé ; ce champ doit donc atteindre `--ctx-size` de l'instance et non un paramètre
d'échantillonnage, sans quoi le plafond serait silencieusement sans effet (risque R2).

**Le plan de contrôle est l'API Ollama elle-même.** Aucun namespace `/admin` n'est créé : la
gestion du catalogue passe par `/api/pull`, `/api/create`, `/api/copy`, `/api/delete` et
`/api/blobs` (mission §24).
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .. import backend
from ..canonical import CanonicalDelta, CanonicalResult, FinishReason, ReasoningBlock, TextBlock
from ..errors import BadRequest, Forbidden, ModelNotFound, NotImplementedByDesign, OllamaError
from ..runtime.capabilities import CAPABILITY_EMBEDDING, detect
from ..service import Service
from ..storage.blobs import BlobError, is_valid_digest
from ..storage.manifests import Artifacts, Manifest, ManifestError, RuntimeConfig
from . import ollama_parse as parse
from . import ollama_serialize as serialize
from .common import get_service, read_body, require_management

NDJSON = "application/x-ndjson"

router = APIRouter()


# --- Découverte ---------------------------------------------------------------------------------


@router.get("/", response_class=PlainTextResponse)
@router.head("/", response_class=PlainTextResponse)
async def root() -> str:
    """Sonde de vie d'Ollama : texte brut « Ollama is running » (`server/routes.go` l. 1859)."""
    return "Ollama is running"


@router.get("/api/version")
@router.head("/api/version")
async def version(request: Request) -> dict[str, str]:
    return {"version": get_service(request).config.reported_version}


@router.get("/api/status")
async def status() -> dict[str, Any]:
    """État du compte cloud. `ollama.cpp` n'a pas de cloud : il l'annonce explicitement."""
    return {"cloud": {"disabled": True, "source": "ollama.cpp"}}


@router.get("/api/tags")
@router.head("/api/tags")
async def tags(request: Request) -> dict[str, Any]:
    """Liste les modèles installés.

    Endpoint de criticité P0 : `ollama-gateway` s'en sert comme **sonde de disponibilité** et
    comme source de son filtrage par clé (`app/servers.py::probe`). Il doit donc répondre 200
    même quand aucun modèle n'est installé, et porter `name` **et** `model` (risque R4).
    """
    service = get_service(request)
    models = []
    for model in service.registry.list():
        capabilities = detect(
            model.manifest,
            props=_props_of(service, model.name),
            metadata=model.metadata,
            has_mmproj=bool(model.manifest.artifacts.mmproj),
        )
        models.append(serialize.list_model_entry(model, capabilities.as_list()))
    return {"models": models}


def _props_of(service: Service, name: str) -> dict[str, Any] | None:
    """`/props` de l'instance si le modèle est résident, sinon `None`.

    Un modèle chargé permet une détection de capacités plus précise ; un modèle au repos se
    contente des artefacts et du GGUF, qui restent des faits observables.
    """
    resident = service.lifecycle.get_resident(name)
    return resident.instance.props if resident is not None and resident.instance else None


@router.post("/api/show")
async def show(request: Request) -> dict[str, Any]:
    """Détaille un modèle : paramètres, template, capacités, métadonnées."""
    service = get_service(request)
    body = await read_body(request)
    name = parse.require_model(body)
    model = service.registry.get(name)

    metadata = model.metadata
    props = _props_of(service, model.name)
    capabilities = detect(
        model.manifest,
        props=props,
        metadata=metadata,
        has_mmproj=bool(model.manifest.artifacts.mmproj),
    )

    # Le template effectif vient de l'instance quand elle tourne — c'est celui que `llama-server`
    # applique réellement —, sinon du manifest, sinon du GGUF (précédence §5.6).
    template = ""
    if props and isinstance(props.get("chat_template"), str):
        template = props["chat_template"]
    elif model.manifest.template:
        template = model.manifest.template
    elif metadata is not None:
        template = metadata.chat_template

    projector_info = None
    if model.manifest.artifacts.mmproj and service.registry.blobs.has(
        model.manifest.artifacts.mmproj
    ):
        from ..gguf import GGUFError, read_metadata

        try:
            projector = read_metadata(
                service.registry.blobs.path(model.manifest.artifacts.mmproj)
            )
            projector_info = projector.public_model_info()
        except (GGUFError, OSError):
            projector_info = None

    return serialize.show_response(
        model,
        capabilities=capabilities.as_list(),
        template=template,
        system=model.manifest.system,
        parameters=serialize.parameters_block(model.manifest.parameters),
        model_info=metadata.public_model_info() if metadata is not None else {},
        projector_info=projector_info,
    )


@router.get("/api/ps")
async def ps(request: Request) -> dict[str, Any]:
    """Liste les modèles réellement résidents.

    Les informations proviennent du cycle de vie réel : aucun état n'est simulé (mission §6).
    """
    service = get_service(request)
    now = dt.datetime.now(dt.timezone.utc)
    models = []
    for resident in service.lifecycle.residents():
        registered = service.registry.try_get(resident.name)
        taille = resident.estimate.total_bytes
        couches = (
            registered.metadata.block_count
            if registered is not None and registered.metadata is not None
            else 0
        )
        models.append(
            serialize.process_model_entry(
                resident,
                size=taille,
                # La part en VRAM est déduite des couches réellement déportées : c'est elle qui
                # détermine la colonne « PROCESSOR » du CLI Ollama.
                size_vram=serialize.vram_bytes(
                    size=taille,
                    gpu_layers=resident.runtime.gpu_layers,
                    block_count=couches,
                ),
                digest=registered.digest if registered else "",
                details=registered.details() if registered else {},
                now=now,
            )
        )
    return {"models": models}


# --- Inférence -----------------------------------------------------------------------------------


async def _run_inference(
    service: Service, body: dict[str, Any], canonical, *, source: str
):
    """Exécute une requête canonique sur l'instance du modèle demandé.

    Renvoie soit une réponse JSON complète, soit un flux NDJSON, selon `stream`.
    """
    keep_alive = canonical.keep_alive

    # `options.num_ctx` (et ses voisins `num_batch`, `num_gpu`, `num_thread`) configurent
    # l'instance, pas l'échantillonnage : ils deviennent des drapeaux `llama-server`. C'est ce
    # qui rend effectif le plafond de contexte injecté par `ollama-gateway` (risque R2) ; les
    # traiter comme des paramètres de génération les rendrait silencieusement inopérants.
    overrides = parse.runtime_overrides(body.get("options"))
    runtime_override = RuntimeConfig(**overrides) if overrides else None

    started = time.monotonic()
    async with service.lifecycle.acquire(
        canonical.model.display_shortest(), keep_alive, runtime_override
    ) as resident:
        # La vérification de capacités a lieu ICI, sur le modèle chargé : `/props` décrit le
        # modèle réellement en mémoire, alors qu'avant chargement on ne dispose que du GGUF, ce
        # qui produit des faux négatifs (un template dont le rendu des outils n'est pas
        # détectable textuellement serait refusé à tort).
        _reject_unsupported(resident, canonical, requested=canonical.model.display_shortest())

        load_s = max(0.0, resident.loaded_at - started) if resident.loaded_at > started else 0.0
        model_name = resident.name
        upstream = backend.serialize_request(canonical, model_id=model_name)
        client = resident.instance.client

        if not canonical.stream:
            result = await backend.chat(client, upstream, model=model_name)
            return _final_payload(source, model_name, result, load_s=load_s, streamed=False)

        return await _stream_inference(client, upstream, model_name, source, load_s)


async def _stream_inference(client, upstream, model_name, source, load_s):
    """Produit le flux NDJSON d'Ollama à partir des deltas canoniques."""

    async def generate() -> AsyncIterator[bytes]:
        content: list[TextBlock | ReasoningBlock] = []
        tool_calls: tuple = ()
        finish = FinishReason.STOP
        usage = None
        timings = None

        async for delta in backend.chat_stream(client, upstream):
            if delta.text:
                content.append(TextBlock(text=delta.text))
            if delta.reasoning:
                content.append(ReasoningBlock(text=delta.reasoning))
            if delta.tool_calls:
                tool_calls = delta.tool_calls
            if delta.usage is not None:
                usage = delta.usage
            if delta.timings is not None:
                timings = delta.timings
            if delta.finish_reason is not None:
                finish = delta.finish_reason

            if delta.text or delta.reasoning or delta.tool_calls:
                chunk = (
                    serialize.chat_chunk(model_name, delta)
                    if source == "chat"
                    else serialize.generate_chunk(model_name, delta)
                )
                yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")

        result = CanonicalResult(
            model=model_name,
            content=tuple(content),
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage or CanonicalResult(model=model_name).usage,
            timings=timings or CanonicalResult(model=model_name).timings,
        )
        final = (
            serialize.chat_stream_final(model_name, result, load_s=load_s)
            if source == "chat"
            else serialize.generate_final(model_name, result, load_s=load_s, streamed=True)
        )
        yield (json.dumps(final, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(generate(), media_type=NDJSON)


def _final_payload(source: str, model_name: str, result: CanonicalResult, *, load_s: float,
                   streamed: bool) -> dict[str, Any]:
    if source == "chat":
        return serialize.chat_final(model_name, result, load_s=load_s)
    return serialize.generate_final(model_name, result, load_s=load_s, streamed=streamed)


@router.post("/api/chat")
async def chat(request: Request):
    """Conversation. Messages, outils, résultats d'outils, images, format, raisonnement."""
    service = get_service(request)
    body = await read_body(request)
    name = parse.require_model(body)
    model = service.registry.get(name)

    # Un `/api/chat` sans message et avec `keep_alive: 0` est la façon documentée de décharger un
    # modèle (`docs/api.md` l. 1141-1149). Aucune inférence n'a alors lieu.
    messages = body.get("messages")
    keep_alive = parse.parse_keep_alive_field(body.get("keep_alive"))
    if not messages and keep_alive is not None and keep_alive.unloads_immediately:
        await service.lifecycle.unload(model.name)
        return _unload_ack(model.name, "chat")
    if not messages and keep_alive is not None:
        await service.lifecycle.ensure_ready(model.name, keep_alive)
        return _load_ack(model.name, "chat")

    canonical = parse.parse_chat_request(body, model.ref)
    return await _run_inference(service, body, canonical, source="chat")


@router.post("/api/generate")
async def generate(request: Request):
    """Complétion simple, avec `system`, `template`, `raw`, `format` et images."""
    service = get_service(request)
    body = await read_body(request)
    name = parse.require_model(body)
    model = service.registry.get(name)

    prompt = body.get("prompt")
    keep_alive = parse.parse_keep_alive_field(body.get("keep_alive"))
    if not prompt and keep_alive is not None and keep_alive.unloads_immediately:
        await service.lifecycle.unload(model.name)
        return _unload_ack(model.name, "generate")
    if not prompt and keep_alive is not None:
        await service.lifecycle.ensure_ready(model.name, keep_alive)
        return _load_ack(model.name, "generate")

    canonical = parse.parse_generate_request(body, model.ref)
    return await _run_inference(service, body, canonical, source="generate")


def _reject_unsupported(resident, canonical, *, requested: str) -> None:
    """Refuse une requête que le modèle ne peut pas honorer, avec le message d'Ollama.

    Les capacités utilisées sont celles détectées au chargement depuis `/props` : ce sont les
    plus fiables, puisqu'elles décrivent le modèle réellement en mémoire (OC-032). Annoncer une
    capacité absente puis échouer à l'exécution serait pire que refuser — le client ne saurait
    pas si c'est lui ou le serveur qui a tort.
    """
    capabilities = resident.capabilities
    if canonical.thinking.is_requested and "thinking" not in capabilities:
        raise BadRequest(f'"{requested}" does not support thinking')
    if canonical.tools and "tools" not in capabilities:
        raise BadRequest(f'"{requested}" does not support tools')
    if any(getattr(message, "images", ()) for message in canonical.messages) and (
        "vision" not in capabilities
    ):
        raise BadRequest(f'"{requested}" does not support vision')


def _unload_ack(model: str, source: str) -> dict[str, Any]:
    """Accusé de déchargement, au format qu'Ollama renvoie dans ce cas (`done_reason: unload`)."""
    payload = {
        "model": model,
        "created_at": serialize.now_rfc3339(),
        "done": True,
        "done_reason": "unload",
    }
    payload["message" if source == "chat" else "response"] = (
        {"role": "assistant", "content": ""} if source == "chat" else ""
    )
    return payload


def _load_ack(model: str, source: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "created_at": serialize.now_rfc3339(),
        "done": True,
        "done_reason": "load",
    }
    payload["message" if source == "chat" else "response"] = (
        {"role": "assistant", "content": ""} if source == "chat" else ""
    )
    return payload


# --- Embeddings ------------------------------------------------------------------------------------


@router.post("/api/embed")
async def embed(request: Request) -> dict[str, Any]:
    """Embeddings, forme plurielle (`api.EmbedRequest`)."""
    service = get_service(request)
    body = await read_body(request)
    name = parse.require_model(body)
    model = service.registry.get(name)

    raw_input = body.get("input")
    if raw_input is None:
        raise BadRequest("input is required")
    inputs = [raw_input] if isinstance(raw_input, str) else raw_input
    if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
        raise BadRequest("input must be a string or an array of strings")

    keep_alive = parse.parse_keep_alive_field(body.get("keep_alive"))
    started = time.monotonic()
    async with service.lifecycle.acquire(model.name, keep_alive) as resident:
        vectors, usage = await backend.embeddings(
            resident.instance.client, model=resident.name, inputs=inputs
        )
    return serialize.embed_response(
        model.name, vectors,
        prompt_eval_count=usage.prompt_eval_count,
        total_s=time.monotonic() - started,
    )


@router.post("/api/embeddings")
async def embeddings_legacy(request: Request) -> dict[str, Any]:
    """Embeddings, forme legacy singulière (`api.EmbeddingRequest`).

    Schéma volontairement différent de `/api/embed` : entrée `prompt`, sortie `{"embedding": [...]}`.
    """
    service = get_service(request)
    body = await read_body(request)
    name = parse.require_model(body)
    model = service.registry.get(name)

    prompt = body.get("prompt")
    if prompt is None:
        prompt = ""
    if not isinstance(prompt, str):
        raise BadRequest("prompt must be a string")

    keep_alive = parse.parse_keep_alive_field(body.get("keep_alive"))
    async with service.lifecycle.acquire(model.name, keep_alive) as resident:
        vectors, _usage = await backend.embeddings(
            resident.instance.client, model=resident.name, inputs=[prompt]
        )
    return serialize.embeddings_response(vectors[0] if vectors else [])


# --- Plan de contrôle -------------------------------------------------------------------------------


@router.post("/api/copy")
async def copy(request: Request) -> Response:
    """Duplique un modèle sous un autre nom. Aucun octet n'est recopié (stockage par contenu)."""
    service = get_service(request)
    require_management(service)
    body = await read_body(request)

    source = str(body.get("source") or "")
    destination = str(body.get("destination") or "")
    if not source or not destination:
        raise BadRequest("source and destination are required")

    service.registry.copy(source, destination)
    return Response(status_code=200)


@router.delete("/api/delete")
async def delete(request: Request) -> Response:
    """Supprime un modèle. 404 si absent — c'est ce qu'attend `ollama-gateway`."""
    service = get_service(request)
    require_management(service)
    body = await read_body(request)

    name = parse.require_model(body)
    if not name:
        raise BadRequest("model is required")

    await service.lifecycle.unload(name)
    if not service.registry.delete(name):
        raise ModelNotFound(name)
    return Response(status_code=200)


@router.head("/api/blobs/{digest}")
async def head_blob(digest: str, request: Request) -> Response:
    """Indique si un blob est déjà présent, pour éviter de le retransmettre."""
    service = get_service(request)
    require_management(service)
    if not is_valid_digest(digest) or not service.registry.blobs.has(digest):
        return Response(status_code=404)
    return Response(status_code=200)


@router.post("/api/blobs/{digest}")
async def create_blob(digest: str, request: Request) -> Response:
    """Téléverse un blob, vérifié contre le digest annoncé.

    Le nom de fichier est dérivé du digest **recalculé localement**, jamais du chemin fourni :
    un digest malformé est refusé avant tout accès disque (risque R8).
    """
    service = get_service(request)
    require_management(service)

    if not is_valid_digest(digest):
        raise BadRequest(f"invalid digest: {digest}")

    async def chunks():
        async for chunk in request.stream():
            if chunk:
                yield chunk

    try:
        collected = [chunk async for chunk in chunks()]
        service.registry.blobs.ingest_chunks(collected, expected_digest=digest)
    except BlobError as exc:
        raise BadRequest(str(exc)) from exc
    return Response(status_code=201)


@router.post("/api/create")
async def create(request: Request):
    """Crée un modèle à partir d'un modèle existant ou de blobs déjà téléversés.

    Sous-ensemble assumé de `api.CreateRequest` : `from`, `files`, `template`, `system`,
    `parameters`, `license`. Les capacités absentes — quantisation à la volée, conversion de
    Safetensors — sont refusées explicitement plutôt que silencieusement ignorées.
    """
    service = get_service(request)
    require_management(service)
    body = await read_body(request)

    name = parse.require_model(body)
    if not name:
        raise BadRequest("model is required")

    if body.get("quantize") or body.get("quantization"):
        raise NotImplementedByDesign(
            "quantization is not supported by ollama.cpp: quantize the GGUF beforehand"
        )

    target = service.registry.resolve(name)
    source_name = body.get("from")
    files = body.get("files")

    if isinstance(source_name, str) and source_name:
        base = service.registry.get(source_name)
        artifacts = base.manifest.artifacts
        runtime = base.manifest.runtime
        source = base.manifest.source
    elif isinstance(files, dict) and files:
        artifacts = _artifacts_from_files(service, files)
        runtime = None
        source = None
    else:
        raise BadRequest("either 'from' or 'files' is required")

    from ..storage.manifests import ModelSource, RuntimeConfig

    manifest = Manifest(
        name=target,
        artifacts=artifacts,
        source=source or ModelSource(type="file", reference=""),
        runtime=runtime or RuntimeConfig(),
        system=str(body.get("system") or ""),
        template=str(body.get("template") or ""),
        license=_license_text(body.get("license")),
        parameters=body.get("parameters") if isinstance(body.get("parameters"), dict) else {},
    )

    try:
        service.registry.install(manifest)
    except ManifestError as exc:
        raise BadRequest(str(exc)) from exc

    stream = body.get("stream")
    if stream is False:
        return {"status": "success"}

    async def progress() -> AsyncIterator[bytes]:
        for status in ("creating manifest", "success"):
            yield (json.dumps({"status": status}) + "\n").encode("utf-8")

    return StreamingResponse(progress(), media_type=NDJSON)


def _artifacts_from_files(service: Service, files: dict[str, Any]) -> Artifacts:
    """Construit les artefacts depuis la table `files` d'`/api/create`.

    Ollama y associe un nom de fichier à un digest de blob déjà téléversé. Le rôle de chaque
    artefact est déduit du nom, `mmproj` étant reconnu par convention.
    """
    model_digest = ""
    mmproj_digest = None

    for filename, digest in files.items():
        if not isinstance(digest, str) or not is_valid_digest(digest):
            raise BadRequest(f"invalid digest for file {filename!r}")
        if not service.registry.blobs.has(digest):
            raise BadRequest(f"blob not found for file {filename!r}: upload it first")
        if "mmproj" in filename.lower() or "projector" in filename.lower():
            mmproj_digest = digest
        elif not model_digest:
            model_digest = digest

    if not model_digest:
        raise BadRequest("no model file provided")
    return Artifacts(model=model_digest, mmproj=mmproj_digest)


def _license_text(raw: Any) -> str:
    """`license` accepte une chaîne ou une liste de chaînes (`CreateRequest.License any`)."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(str(item) for item in raw)
    return ""


@router.post("/api/pull")
async def pull(request: Request):
    """Télécharge un modèle depuis une source configurée.

    `ollama-gateway` appelle cet endpoint en `stream: false` et exige `{"status": "success"}` en
    200 ; toute autre valeur de `status` est traitée comme un échec (`app/servers.py` l. 402-433).
    """
    service = get_service(request)
    require_management(service)
    body = await read_body(request)

    name = parse.require_model(body)
    if not name:
        raise BadRequest("model is required")

    from ..sources import PullError, pull_model

    stream = body.get("stream")
    if stream is False:
        try:
            await _drain(pull_model(service, name))
        except PullError as exc:
            raise BadRequest(str(exc)) from exc
        return {"status": "success"}

    async def progress() -> AsyncIterator[bytes]:
        try:
            async for event in pull_model(service, name):
                yield (json.dumps(event) + "\n").encode("utf-8")
        except PullError as exc:
            # Une erreur survenue en cours de flux ne peut plus changer le code HTTP : Ollama
            # émet alors un objet `{"error": ...}` dans le flux, ce que l'on reproduit.
            yield (json.dumps({"error": str(exc)}) + "\n").encode("utf-8")

    return StreamingResponse(progress(), media_type=NDJSON)


async def _drain(iterator) -> None:
    async for _event in iterator:
        pass


@router.post("/api/push")
async def push() -> Response:
    """Hors périmètre assumé : pas de registre de publication (architecture §5.3).

    Répond 501 avec un message explicite, jamais un 404 de routeur — un 404 ferait croire à
    `ollama-gateway` que l'endpoint n'existe pas, alors qu'il est délibérément refusé (risque R1).
    """
    raise NotImplementedByDesign(
        "pushing models is not supported by ollama.cpp: there is no upstream model registry"
    )
