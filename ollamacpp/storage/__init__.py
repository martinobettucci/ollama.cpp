"""Stockage de `ollama.cpp` : blobs adressés par contenu et manifests.

@spec docs/BACKLOG.md OC-020 « BlobStore adressé par contenu », OC-021 « Manifests de modèles »
@spec docs/ollama.cpp-architecture.md §5.10 « Arborescence de données »
@spec docs/DAT.md §4 « Modèle de données »
"""

from .blobs import BlobError, BlobInfo, BlobStore, compute_digest, is_valid_digest
from .manifests import (
    Artifacts,
    LifecycleConfig,
    Manifest,
    ManifestError,
    ManifestStore,
    ModelSource,
    RuntimeConfig,
    SCHEMA_VERSION,
)

__all__ = [
    "Artifacts",
    "BlobError",
    "BlobInfo",
    "BlobStore",
    "LifecycleConfig",
    "Manifest",
    "ManifestError",
    "ManifestStore",
    "ModelSource",
    "RuntimeConfig",
    "SCHEMA_VERSION",
    "compute_digest",
    "is_valid_digest",
]
