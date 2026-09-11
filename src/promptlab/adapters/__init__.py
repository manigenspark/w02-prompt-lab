"""Model adapters for local Ollama runtimes."""

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import OllamaAdapter

__all__ = [
    "CompletionRequest",
    "CompletionResult",
    "ModelAdapter",
    "OllamaAdapter",
]
