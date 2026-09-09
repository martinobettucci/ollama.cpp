"""Stockage d'artefacts adressé par contenu.

@spec docs/BACKLOG.md OC-020 « BlobStore adressé par contenu »
@spec docs/ollama.cpp-architecture.md §5.10 « Arborescence de données », §8 risques R4 et R8
@spec docs/DAT.md §4.1 « Arborescence », §7 « Sécurité »

Le layout est inspiré de celui d'Ollama (`blobs/sha256-<hex>`) parce que les propriétés
recherchées — déduplication, checksums, artefacts partagés, installation atomique — viennent du
**layout**, pas du protocole de registre. Le protocole OCI d'Ollama n'est pas repris
(`docs/ollama.cpp-architecture.md` §5.3).

**Risque R8 — injection de chemin.** Aucun nom de fichier ne provient jamais d'un manifest
distant : le nom d'un blob est **dérivé du digest calculé localement**. Un manifest hostile
contenant `../../etc/passwd` ne peut donc pas désigner un chemin d'écriture. `digest_to_filename`
valide en outre la forme du digest avant tout accès disque.

**Installation atomique.** Un artefact est écrit dans `tmp/`, vérifié, puis déplacé par
`os.replace` — atomique sur le même système de fichiers. Une interruption laisse au pire un
fichier `.partial` collecté au démarrage, jamais un blob tronqué présenté comme valide.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

#: Un digest est `sha256:<64 caractères hexadécimaux minuscules>`. Forme close, validée avant
#: toute utilisation comme composant de chemin.
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_CHUNK_SIZE = 1024 * 1024


class BlobError(ValueError):
    """Digest malformé, artefact absent ou checksum non conforme."""


def is_valid_digest(digest: str) -> bool:
    return bool(DIGEST_PATTERN.match(digest))


def digest_to_filename(digest: str) -> str:
    """Convertit `sha256:<hex>` en nom de fichier `sha256-<hex>`.

    Le `:` est remplacé par `-` comme chez Ollama : c'est un caractère interdit dans un nom de
    fichier sous Windows. La validation préalable garantit qu'aucun séparateur de chemin ni
    séquence `..` ne peut survivre à la conversion (risque R8).
    """
    if not is_valid_digest(digest):
        raise BlobError(f"digest invalide : {digest!r}")
    return digest.replace(":", "-", 1)


def filename_to_digest(filename: str) -> str:
    """Inverse de `digest_to_filename`. Lève si le nom n'est pas un blob valide."""
    digest = filename.replace("-", ":", 1)
    if not is_valid_digest(digest):
        raise BlobError(f"nom de blob invalide : {filename!r}")
    return digest


def compute_digest(source: Path | BinaryIO) -> str:
    """Calcule le digest SHA-256 d'un fichier ou d'un flux, par blocs (aucun chargement complet)."""
    hasher = hashlib.sha256()
    if isinstance(source, Path):
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                hasher.update(chunk)
    else:
        for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True, slots=True)
class BlobInfo:
    digest: str
    size: int


class BlobStore:
    """Magasin d'artefacts adressé par contenu."""

    def __init__(self, blobs_dir: Path, tmp_dir: Path) -> None:
        self._blobs_dir = Path(blobs_dir)
        self._tmp_dir = Path(tmp_dir)

    @property
    def blobs_dir(self) -> Path:
        return self._blobs_dir

    def ensure_layout(self) -> None:
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Lecture ---------------------------------------------------------------------------------

    def path(self, digest: str) -> Path:
        """Chemin d'un blob. Le digest est validé avant d'être utilisé comme nom de fichier."""
        return self._blobs_dir / digest_to_filename(digest)

    def has(self, digest: str) -> bool:
        try:
            return self.path(digest).is_file()
        except BlobError:
            return False

    def size(self, digest: str) -> int:
        path = self.path(digest)
        if not path.is_file():
            raise BlobError(f"blob absent : {digest}")
        return path.stat().st_size

    def info(self, digest: str) -> BlobInfo:
        return BlobInfo(digest=digest, size=self.size(digest))

    def list(self) -> Iterator[BlobInfo]:
        """Énumère les blobs présents, en ignorant les fichiers étrangers au layout."""
        if not self._blobs_dir.is_dir():
            return
        for entry in sorted(self._blobs_dir.iterdir()):
            if not entry.is_file():
                continue
            try:
                digest = filename_to_digest(entry.name)
            except BlobError:
                continue
            yield BlobInfo(digest=digest, size=entry.stat().st_size)

    # --- Écriture --------------------------------------------------------------------------------

    def ingest_bytes(self, data: bytes, expected_digest: str | None = None) -> BlobInfo:
        return self.ingest_chunks([data], expected_digest)

    def ingest_file(self, source: Path, expected_digest: str | None = None) -> BlobInfo:
        """Copie un fichier local dans le magasin. Le fichier source n'est pas modifié."""
        with Path(source).open("rb") as stream:
            return self.ingest_chunks(iter(lambda: stream.read(_CHUNK_SIZE), b""), expected_digest)

    def ingest_chunks(
        self, chunks: Iterable[bytes], expected_digest: str | None = None
    ) -> BlobInfo:
        """Écrit un flux de blocs, vérifie le checksum, puis installe atomiquement.

        Si `expected_digest` est fourni et ne correspond pas au contenu reçu, le fichier temporaire
        est **détruit** et rien n'est installé : un artefact corrompu ou substitué ne doit jamais
        atterrir dans le magasin.

        La déduplication est implicite : un contenu déjà présent produit le même digest, donc le
        même chemin. On évite alors le déplacement plutôt que de réécrire par-dessus un blob que
        d'autres manifests référencent peut-être déjà.
        """
        if expected_digest is not None and not is_valid_digest(expected_digest):
            raise BlobError(f"digest attendu invalide : {expected_digest!r}")

        self.ensure_layout()
        temp_path = self._tmp_dir / f"{uuid.uuid4().hex}.partial"
        hasher = hashlib.sha256()
        size = 0

        try:
            with temp_path.open("wb") as stream:
                for chunk in chunks:
                    hasher.update(chunk)
                    size += len(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())

            digest = f"sha256:{hasher.hexdigest()}"
            if expected_digest is not None and digest != expected_digest:
                raise BlobError(
                    f"checksum non conforme : attendu {expected_digest}, obtenu {digest}"
                )

            destination = self.path(digest)
            if destination.exists():
                temp_path.unlink(missing_ok=True)
            else:
                os.replace(temp_path, destination)

            return BlobInfo(digest=digest, size=size)
        finally:
            # Le `finally` couvre aussi l'échec de checksum : aucun `.partial` ne survit à une
            # ingestion refusée.
            temp_path.unlink(missing_ok=True)

    # --- Suppression et entretien -----------------------------------------------------------------

    def delete(self, digest: str) -> bool:
        """Supprime un blob. Renvoie `False` s'il était déjà absent (opération idempotente)."""
        path = self.path(digest)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def sweep_tmp(self) -> int:
        """Collecte les téléchargements interrompus au démarrage. Renvoie le nombre supprimé."""
        if not self._tmp_dir.is_dir():
            return 0
        removed = 0
        for entry in self._tmp_dir.iterdir():
            if entry.is_file() and entry.name.endswith(".partial"):
                entry.unlink(missing_ok=True)
                removed += 1
            elif entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        return removed

    def collect_garbage(self, referenced: set[str]) -> list[str]:
        """Supprime les blobs qu'aucun manifest ne référence plus.

        L'appelant fournit l'ensemble des digests référencés : le magasin ne connaît pas les
        manifests, et cette séparation évite un couplage circulaire entre stockage et registre.
        """
        removed: list[str] = []
        for blob in list(self.list()):
            if blob.digest not in referenced:
                self.delete(blob.digest)
                removed.append(blob.digest)
        return removed
