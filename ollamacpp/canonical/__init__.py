"""Représentation conversationnelle canonique de `ollama.cpp`.

@spec docs/BACKLOG.md OC-014 « Représentation conversationnelle canonique »
@spec docs/ollama.cpp-architecture.md §5.4 « Modèle conversationnel canonique »

Point d'entrée unique du paquet : les façades importent depuis `ollamacpp.canonical`, jamais
depuis les sous-modules, afin que la frontière du contrat interne reste visible.
"""

from .messages import (
    AssistantMessage,
    CanonicalMessage,
    ContentBlock,
    ImageInput,
    ReasoningBlock,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    link_tool_results,
)
from .request import (
    CanonicalRequest,
    ResponseFormat,
    ResponseFormatKind,
    SamplingOptions,
    SourceAPI,
    ThinkingLevel,
    ThinkingRequest,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
)
from .result import CanonicalDelta, CanonicalResult, FinishReason, Timings, Usage

__all__ = [
    "AssistantMessage",
    "CanonicalDelta",
    "CanonicalMessage",
    "CanonicalRequest",
    "CanonicalResult",
    "ContentBlock",
    "FinishReason",
    "ImageInput",
    "ReasoningBlock",
    "ResponseFormat",
    "ResponseFormatKind",
    "SamplingOptions",
    "SourceAPI",
    "SystemMessage",
    "TextBlock",
    "ThinkingLevel",
    "ThinkingRequest",
    "Timings",
    "ToolCall",
    "ToolChoice",
    "ToolChoiceMode",
    "ToolDefinition",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "link_tool_results",
]
