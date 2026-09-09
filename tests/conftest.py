"""Fixtures partagées.

@verifies docs/BACKLOG.md OC-030 « LlamaServerSupervisor », OC-033 « ModelLifecycleManager »,
          OC-034 « ModelScheduler »
@verifies docs/ollama.cpp-architecture.md §5.1 « Vue d'ensemble »
@verifies docs/DAT.md §2 « Services et processus », §13 « Données de développement »

Le faux `llama-server` est un script exécutable, lancé par le **vrai** superviseur, exactement
comme le serait le binaire natif : aucun code de production n'est modifié ni contourné pour les
tests. Les tests d'intégration exercent donc réellement l'allocation de port, l'attente de
`/health`, la lecture de `/props`, la détection de capacités, le proxy de streaming et l'arrêt
par signal. Seule l'inférence est déterministe (cf. `tests/fake_llama_server.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ollamacpp.config import Config
from ollamacpp.names import parse
from ollamacpp.registry import ModelRegistry
from ollamacpp.runtime import LlamaServerSupervisor, ModelLifecycleManager, ModelScheduler
from ollamacpp.storage import Artifacts, BlobStore, LifecycleConfig, Manifest, ManifestStore, RuntimeConfig

from .ggufbuild import DEFAULT_KV, build_gguf, build_mmproj

#: Chemin du faux `llama-server`. Exécutable, avec un shebang : le superviseur le lance comme
#: n'importe quel binaire, sans traitement particulier.
FAKE_LLAMA_SERVER = str(Path(__file__).parent / "fake_llama_server.py")


@pytest.fixture
def config(tmp_path) -> Config:
    """Configuration de test : stockage isolé, faux binaire amont, plage de ports dédiée."""
    return Config(
        models_dir=tmp_path / "models",
        llama_server_bin=FAKE_LLAMA_SERVER,
        llama_server_port_min=19100,
        llama_server_port_max=19399,
        load_timeout_s=20.0,
        request_timeout_s=20.0,
        max_loaded_models=2,
        # Budget explicite et marge nulle : les décisions du scheduler ne doivent pas dépendre de
        # la RAM de la machine de test, sinon la suite passerait ou échouerait selon l'hôte.
        # Les tests de pression mémoire posent leur propre budget.
        memory_limit_bytes=64 * 1024 ** 3,
        memory_safety_margin=0.0,
        default_context=4096,
    )


@pytest.fixture
def registry(config: Config) -> ModelRegistry:
    config.ensure_layout()
    blobs = BlobStore(config.blobs_dir, config.tmp_dir)
    blobs.ensure_layout()
    return ModelRegistry(blobs, ManifestStore(config.manifests_dir))


@pytest.fixture
def supervisor(config: Config) -> LlamaServerSupervisor:
    return LlamaServerSupervisor(
        binary=config.llama_server_bin,
        host=config.llama_server_host,
        port_min=config.llama_server_port_min,
        port_max=config.llama_server_port_max,
        load_timeout_s=config.load_timeout_s,
        request_timeout_s=config.request_timeout_s,
        default_context=config.default_context,
    )


@pytest.fixture
def scheduler(config: Config) -> ModelScheduler:
    return ModelScheduler(
        memory_budget_bytes=config.memory_limit_bytes,
        max_loaded_models=config.max_loaded_models,
    )


@pytest.fixture
async def lifecycle(config, registry, supervisor, scheduler):
    """Gestionnaire de cycle de vie complet, branché sur le faux `llama-server`.

    L'arrêt est garanti par la fixture : aucun processus ne doit survivre à un test, même en cas
    d'échec — sinon les ports resteraient pris et les tests suivants échoueraient sans rapport
    avec leur objet.
    """
    manager = ModelLifecycleManager(
        registry=registry,
        supervisor=supervisor,
        scheduler=scheduler,
        config=config,
    )
    try:
        yield manager
    finally:
        await manager.shutdown()


def install_model(
    registry: ModelRegistry,
    name: str = "qwen3:8b",
    *,
    with_mmproj: bool = False,
    runtime: RuntimeConfig | None = None,
    lifecycle_config: LifecycleConfig | None = None,
    kv: dict | None = None,
) -> Manifest:
    """Installe un modèle de test complet, avec de vrais artefacts GGUF dans le magasin."""
    model_blob = registry.blobs.ingest_bytes(build_gguf(kv or DEFAULT_KV))
    mmproj_digest = registry.blobs.ingest_bytes(build_mmproj()).digest if with_mmproj else None

    manifest = Manifest(
        name=parse(name),
        artifacts=Artifacts(model=model_blob.digest, mmproj=mmproj_digest),
        runtime=runtime or RuntimeConfig(),
        lifecycle=lifecycle_config or LifecycleConfig(),
    )
    registry.install(manifest)
    return manifest


@pytest.fixture(autouse=True)
def fake_server_env(monkeypatch):
    """Neutralise les variables de simulation du faux serveur entre deux tests."""
    for name in ("FAKE_LLAMA_EXIT_CODE", "FAKE_LLAMA_START_DELAY", "FAKE_LLAMA_VISION",
                 "FAKE_LLAMA_TOOLS", "FAKE_LLAMA_THINKING", "FAKE_LLAMA_LOG_BYTES"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(config):
    """Client HTTP sur l'application complète, adossée au faux `llama-server`.

    Le cycle de vie ASGI est exécuté : le service démarre réellement, lance de vrais processus à
    la demande, et les arrête à la fermeture. Ce sont donc des tests d'API de bout en bout, pas
    des appels de fonctions.
    """
    import warnings

    from fastapi.testclient import TestClient

    from ollamacpp.app import create_app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(create_app(config)) as test_client:
            yield test_client


@pytest.fixture
def installed(client, config):
    """Registre du service démarré, pour installer des modèles vus par l'application."""
    return client.app.state.service.registry
