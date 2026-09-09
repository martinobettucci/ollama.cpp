"""Registre de modèles de `ollama.cpp`.

@spec docs/BACKLOG.md OC-022 « ModelRegistry »
@spec docs/ollama.cpp-architecture.md §6 « Interfaces internes proposées »
"""

from .registry import ModelRegistry, RegisteredModel

__all__ = ["ModelRegistry", "RegisteredModel"]
