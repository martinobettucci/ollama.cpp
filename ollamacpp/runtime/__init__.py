"""Runtime de `ollama.cpp` : supervision, capacités, cycle de vie, ordonnancement.

@spec docs/BACKLOG.md OC-030 à OC-035
@spec docs/ollama.cpp-architecture.md §5.7 « États du cycle de vie », §5.8 « Scheduler »
@spec docs/DAT.md §2 « Services et processus »
"""

from .args import build_args, filter_extra_args
from .capabilities import DetectedCapabilities, context_length, detect
from .lifecycle import ModelLifecycleManager, ModelState, Resident
from .memory import (
    FixedMemoryProbe,
    HostMemoryProbe,
    MemoryEstimate,
    estimate_kv_bytes,
    estimate_model_memory,
    resolve_memory_budget,
)
from .scheduler import AdmissionPlan, EvictionDecision, ModelScheduler, ResidentInfo
from .supervisor import LlamaServerInstance, LlamaServerSupervisor, allocate_port

__all__ = [
    "AdmissionPlan",
    "DetectedCapabilities",
    "EvictionDecision",
    "FixedMemoryProbe",
    "HostMemoryProbe",
    "LlamaServerInstance",
    "LlamaServerSupervisor",
    "MemoryEstimate",
    "ModelLifecycleManager",
    "ModelScheduler",
    "ModelState",
    "Resident",
    "ResidentInfo",
    "allocate_port",
    "build_args",
    "context_length",
    "detect",
    "estimate_kv_bytes",
    "estimate_model_memory",
    "filter_extra_args",
    "resolve_memory_budget",
]
