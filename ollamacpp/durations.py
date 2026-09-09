"""Durées : `keep_alive` (sémantique Ollama) et durées de réponse en nanosecondes.

@spec docs/BACKLOG.md OC-013 « Durées et keep_alive »
@spec docs/ollama.cpp-architecture.md §2.2 « Conventions comportementales », §8 risques R3 et R5
@spec docs/DAT.md §1 « Composants » (module `ollamacpp/durations.py`)

Deux pièges de compatibilité sont traités ici, isolés dans un module dédié précisément parce
qu'ils sont silencieux quand on se trompe.

**R3 — toutes les durées des réponses Ollama sont en nanosecondes** (`docs/api.md`, section
« Conventions »). Un client calcule des tokens/s en divisant `eval_count` par `eval_duration`
puis en multipliant par 10⁹ : une durée émise en secondes produit un débit faux d'un facteur 10⁹
sans lever la moindre erreur.

**R5 — `keep_alive` a une sémantique à quatre branches**, portée fidèlement depuis
`Duration.UnmarshalJSON` (`api/types.go` l. 1243-1271, révision auditée `d67ad83`) :

| Valeur JSON reçue | Interprétation |
|---|---|
| absente ou `null`  | 5 minutes (défaut Ollama) |
| nombre `n >= 0`    | `n` **secondes** (et non nanosecondes ni millisecondes) |
| nombre `n < 0`     | résidence illimitée |
| chaîne             | durée Go (`"10m"`, `"1h30m"`, `"300ms"`) |
| chaîne négative    | résidence illimitée |
| tout autre type    | erreur de requête |

Le cas `0` n'est pas une branche séparée : c'est un `n >= 0` qui vaut zéro seconde, donc un
déchargement dès la fin de la requête.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

#: Durée de résidence par défaut quand la requête ne précise rien (`docs/api.md`).
DEFAULT_KEEP_ALIVE_SECONDS = 300.0

_NANOSECONDS_PER_SECOND = 1_000_000_000

# Unités acceptées par `time.ParseDuration` de Go, en secondes.
_GO_UNITS = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,  # micro sign
    "μs": 1e-6,  # greek small letter mu
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}

_GO_TERM = re.compile(r"([0-9]*\.?[0-9]+)(ns|us|µs|μs|ms|s|m|h)")


class DurationError(ValueError):
    """Valeur de durée non interprétable, à convertir en 400 par la façade appelante."""


@dataclass(frozen=True, slots=True)
class KeepAlive:
    """Durée de résidence demandée pour un modèle.

    `seconds` vaut `math.inf` pour une résidence illimitée et `0.0` pour un déchargement dès la
    fin de la requête. Les deux extrêmes sont des valeurs légitimes, pas des cas d'erreur.
    """

    seconds: float

    @property
    def is_infinite(self) -> bool:
        return math.isinf(self.seconds)

    @property
    def unloads_immediately(self) -> bool:
        return self.seconds <= 0.0

    def __str__(self) -> str:
        return "infinite" if self.is_infinite else f"{self.seconds:g}s"


#: Valeur appliquée en l'absence de `keep_alive` dans la requête.
DEFAULT_KEEP_ALIVE = KeepAlive(DEFAULT_KEEP_ALIVE_SECONDS)

#: Résidence illimitée (`keep_alive` négatif, quel que soit le format).
INFINITE_KEEP_ALIVE = KeepAlive(math.inf)


def parse_go_duration(text: str) -> float:
    """Analyse une durée au format Go et renvoie des secondes.

    Formats acceptés : suite de termes `<nombre><unité>` éventuellement précédée d'un signe,
    plus le cas particulier `"0"` sans unité. Une chaîne vide, une unité inconnue ou un reste non
    consommé lèvent `DurationError` — jamais de repli silencieux sur une valeur par défaut.
    """
    raw = text.strip()
    if not raw:
        raise DurationError("durée vide")

    sign = 1.0
    if raw[0] in "+-":
        if raw[0] == "-":
            sign = -1.0
        raw = raw[1:]

    if raw in {"0", "0.0"}:
        return 0.0

    total = 0.0
    position = 0
    while position < len(raw):
        match = _GO_TERM.match(raw, position)
        if match is None:
            raise DurationError(f"durée invalide : {text!r}")
        total += float(match.group(1)) * _GO_UNITS[match.group(2)]
        position = match.end()

    return sign * total


def parse_keep_alive(value: object) -> KeepAlive:
    """Interprète le champ `keep_alive` d'une requête Ollama.

    `None` couvre à la fois l'absence du champ et un `null` explicite : Ollama laisse dans les
    deux cas le pointeur nul, donc la valeur par défaut s'applique.
    """
    if value is None:
        return DEFAULT_KEEP_ALIVE

    if isinstance(value, bool):
        # `bool` est un `int` en Python mais pas un nombre pour Ollama : refusé explicitement,
        # sans quoi `true` deviendrait silencieusement une seconde.
        raise DurationError("keep_alive doit être un nombre de secondes ou une durée")

    if isinstance(value, (int, float)):
        if math.isnan(value):
            raise DurationError("keep_alive invalide")
        return INFINITE_KEEP_ALIVE if value < 0 else KeepAlive(float(value))

    if isinstance(value, str):
        seconds = parse_go_duration(value)
        return INFINITE_KEEP_ALIVE if seconds < 0 else KeepAlive(seconds)

    raise DurationError(f"keep_alive de type non supporté : {type(value).__name__}")


def seconds_to_nanoseconds(seconds: float) -> int:
    """Convertit des secondes en nanosecondes entières, format des durées de réponse (R3)."""
    return int(seconds * _NANOSECONDS_PER_SECOND)


def format_go_duration(seconds: float) -> str:
    """Formate des secondes comme `time.Duration.String()` de Go.

    Utilisé pour réémettre un `keep_alive` dans une réponse ou une trace, au format que les
    clients Ollama savent relire.
    """
    if math.isinf(seconds):
        return "infinite"
    if seconds == 0:
        return "0s"

    sign = "-" if seconds < 0 else ""
    remaining = abs(seconds)

    if remaining < 1e-6:
        return f"{sign}{remaining * 1e9:g}ns"
    if remaining < 1e-3:
        return f"{sign}{remaining * 1e6:g}µs"
    if remaining < 1:
        return f"{sign}{remaining * 1e3:g}ms"

    hours, rest = divmod(remaining, 3600)
    minutes, secs = divmod(rest, 60)
    out = ""
    if hours:
        out += f"{int(hours)}h"
    if minutes or hours:
        out += f"{int(minutes)}m"
    out += f"{secs:g}s"
    return sign + out
