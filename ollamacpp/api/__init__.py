"""Façades HTTP de `ollama.cpp`.

@spec docs/BACKLOG.md OC-040 à OC-055, OC-070 à OC-076
@spec docs/ollama.cpp-architecture.md §5.1 « Vue d'ensemble »
@spec docs/DAT.md §5.1 « Interfaces exposées »

Quatre façades, un seul runtime. Chacune convertit vers la représentation canonique et jamais
vers une autre façade.
"""

from . import anthropic, ollama, openai, responses

__all__ = ["anthropic", "ollama", "openai", "responses"]
