class UnknownModelError(ValueError):
    """Raised when a model identifier is not in the configured model table."""


class TransientProviderError(Exception):
    """Timeout, connection failure, or temporary Ollama/server failure."""


class PermanentProviderError(Exception):
    """Malformed request, unavailable model, unsupported parameter, or other non-retryable
    request failure."""


class TruncatedResponseError(Exception):
    """Ollama reported that the output token ceiling was reached."""