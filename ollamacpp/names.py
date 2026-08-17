"""Nommage des modèles, compatible Ollama.

@spec docs/BACKLOG.md OC-012 « Nommage des modèles »
@spec docs/ollama.cpp-architecture.md §2.5 « Nommage », §5.10 « Arborescence de données »
@spec docs/DAT.md §1 « Composants » (module `ollamacpp/names.py`)

Portage fidèle de la sémantique de `types/model/name.go` d'Ollama (révision auditée `d67ad83`).
Les règles reproduites sont volontairement identiques, y compris dans leurs cas limites, car la
façade Ollama doit préserver les noms attendus par les clients (mission §27).

Règles reproduites :

- forme `[host/][namespace/]model[:tag]` ;
- hôte par défaut `registry.ollama.ai`, namespace par défaut `library`, tag par défaut `latest` ;
- le découpage du tag n'a lieu que si le dernier `:` suit le dernier `/` (un `:` dans un hôte
  avec port n'est donc pas un tag) ;
- une partie vide autour d'un séparateur présent devient `!MISSING!`, ce qui rend le nom invalide
  au lieu de le faire silencieusement retomber sur une valeur par défaut ;
- `display_shortest()` omet l'hôte quand il vaut le défaut, et le namespace quand hôte et
  namespace valent tous deux les défauts, mais force toujours `model:tag`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

DEFAULT_HOST = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
DEFAULT_TAG = "latest"

#: Marqueur d'une partie promise par un séparateur mais absente (`types/model/name.go` l. 36).
MISSING_PART = "!MISSING!"

_MAX_LEN = {"host": 350, "namespace": 80, "model": 80, "tag": 80}

# Premier caractère : alphanumérique ou `_`. Ensuite : alphanumérique, `_`, `-`, et selon la
# partie `.` (interdit dans un namespace) et `:` (autorisé seulement dans un hôte).
_FIRST = re.compile(r"[A-Za-z0-9_]")
_REST = {
    "host": re.compile(r"[A-Za-z0-9_.:-]"),
    "namespace": re.compile(r"[A-Za-z0-9_-]"),
    "model": re.compile(r"[A-Za-z0-9_.-]"),
    "tag": re.compile(r"[A-Za-z0-9_.-]"),
}


def _is_valid_part(kind: str, value: str) -> bool:
    """Reproduit `isValidPart` : longueur bornée, premier caractère restreint, reste selon la partie."""
    if not 1 <= len(value) <= _MAX_LEN[kind]:
        return False
    if not _FIRST.fullmatch(value[0]):
        return False
    rest = _REST[kind]
    return all(rest.fullmatch(c) for c in value[1:])


def _cut_last(value: str, sep: str) -> tuple[str, str, bool]:
    index = value.rfind(sep)
    if index < 0:
        return value, "", False
    return value[:index], value[index + len(sep):], True


def _cut_promised(value: str, sep: str) -> tuple[str, str, bool]:
    """Reproduit `cutPromised` : un séparateur présent « promet » deux parties non vides."""
    before, after, found = _cut_last(value, sep)
    if not found:
        return before, after, False
    return (before or MISSING_PART), (after or MISSING_PART), True


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Référence de modèle normalisée. Immuable : sert de clé de registre et de cache."""

    host: str = ""
    namespace: str = ""
    model: str = ""
    tag: str = ""

    # --- Validité ----------------------------------------------------------------------------

    def is_fully_qualified(self) -> bool:
        return all(
            _is_valid_part(kind, value)
            for kind, value in (
                ("host", self.host),
                ("namespace", self.namespace),
                ("model", self.model),
                ("tag", self.tag),
            )
        )

    @property
    def is_valid(self) -> bool:
        return self.is_fully_qualified()

    # --- Rendu -------------------------------------------------------------------------------

    def __str__(self) -> str:
        """Reproduit `Name.String()` : concatène les parties non vides, sans omission."""
        out = ""
        if self.host:
            out += f"{self.host}/"
        if self.namespace:
            out += f"{self.namespace}/"
        out += self.model
        if self.tag:
            out += f":{self.tag}"
        return out

    def display_shortest(self) -> str:
        """Nom court tel qu'affiché par les clients Ollama (`Name.DisplayShortest()`).

        L'hôte est omis quand il vaut le défaut ; le namespace n'est omis que si l'hôte l'est
        aussi. `model:tag` est toujours présent, tag compris.
        """
        out = ""
        if self.host.lower() != DEFAULT_HOST.lower():
            out += f"{self.host}/{self.namespace}/"
        elif self.namespace.lower() != DEFAULT_NAMESPACE.lower():
            out += f"{self.namespace}/"
        return f"{out}{self.model}:{self.tag}"

    def filepath(self) -> PurePosixPath:
        """Chemin canonique `{host}/{namespace}/{model}/{tag}` du manifest.

        Refuse une référence non pleinement qualifiée : aucun chemin ne doit pouvoir être dérivé
        d'un nom partiellement valide (risque R8, injection de chemin).
        """
        if not self.is_fully_qualified():
            raise ValueError(f"référence de modèle invalide : {self!s}")
        return PurePosixPath(self.host, self.namespace, self.model, self.tag)

    def equal_fold(self, other: "ModelRef") -> bool:
        return (
            self.host.lower() == other.host.lower()
            and self.namespace.lower() == other.namespace.lower()
            and self.model.lower() == other.model.lower()
            and self.tag.lower() == other.tag.lower()
        )


def parse_bare(name: str) -> ModelRef:
    """Analyse sans application des valeurs par défaut (`ParseNameBare`)."""
    host = namespace = model = tag = ""
    rest = name

    # « / » est illégal dans un tag : si le dernier « : » suit le dernier « / », c'est un tag.
    # C'est ce qui évite de prendre le port d'un hôte (`registre.local:8443/ns/m`) pour un tag.
    if rest.rfind(":") > rest.rfind("/"):
        rest, tag, _ = _cut_promised(rest, ":")

    rest, model, promised = _cut_promised(rest, "/")
    if not promised:
        return ModelRef(model=rest, tag=tag)

    rest, namespace, promised = _cut_promised(rest, "/")
    if not promised:
        return ModelRef(namespace=rest, model=model, tag=tag)

    # Un schéma de protocole éventuel est retiré : seul l'hôte est conservé.
    scheme, sep, after = rest.partition("://")
    host = after if sep else scheme

    return ModelRef(host=host, namespace=namespace, model=model, tag=tag)


def parse(name: str) -> ModelRef:
    """Analyse avec application des valeurs par défaut (`ParseName`).

    Le résultat n'est pas garanti valide : utiliser `ModelRef.is_valid`. Cette distinction est
    reprise telle quelle d'Ollama, car elle permet de renvoyer l'erreur « nom invalide » plutôt
    que de résoudre silencieusement un nom malformé.
    """
    bare = parse_bare(name)
    return replace(
        bare,
        host=bare.host or DEFAULT_HOST,
        namespace=bare.namespace or DEFAULT_NAMESPACE,
        tag=bare.tag or DEFAULT_TAG,
    )


def parse_from_filepath(path: str) -> ModelRef:
    """Analyse un chemin `{host}/{namespace}/{model}/{tag}` (`ParseNameFromFilepath`).

    Renvoie une référence vide si le chemin n'a pas exactement quatre segments ou n'est pas
    pleinement qualifié : un manifest dont le chemin est malformé est ignoré, jamais deviné.
    """
    parts = PurePosixPath(path).parts
    if len(parts) != 4:
        return ModelRef()
    ref = ModelRef(host=parts[0], namespace=parts[1], model=parts[2], tag=parts[3])
    return ref if ref.is_fully_qualified() else ModelRef()
