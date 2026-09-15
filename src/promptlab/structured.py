from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, ModelAdapter


def _unwrap_json(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _coerce(payload: object) -> object:
    if isinstance(payload, list):
        return [_coerce(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    if "text" in payload and "value" not in payload:
        payload["value"] = payload.pop("text")
    citation = payload.get("citation")
    if isinstance(citation, list):
        payload["citation"] = citation[0] if citation else None
    if payload.get("status") == "absent" and "value" not in payload:
        payload["value"] = None
        payload.setdefault("citation", None)
    return {key: _coerce(value) for key, value in payload.items()}


def complete_structured[T: BaseModel](
    adapter: ModelAdapter,
    request: CompletionRequest,
    schema: type[T],
    run_id: str,
    max_repairs: int = 1,
) -> T:
    """Return a schema-validated completion with a bounded semantic repair loop.

    Transport retry remains inside the adapter.
    Schema/content repair belongs here.

    On validation failure, send the validation error text back to the model and
    instruct it to correct only what the error concerns. Do not perform more
    than max_repairs semantic repair attempts.
    """

    current = request
    last_error = ""

    for attempt in range(max_repairs + 1):
        result = adapter.complete(current, run_id)
        if result.text is None:
            raise ValueError(result.error_type or "empty model response")
        if not result.succeeded and result.error_type != "TruncatedResponseError":
            raise ValueError(result.error_type or "empty model response")

        try:
            payload: object = _coerce(json.loads(_unwrap_json(result.text)))
            if isinstance(payload, dict):
                allowed = set(schema.model_fields)
                payload = {key: value for key, value in payload.items() if key in allowed}
            return schema.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            if attempt >= max_repairs:
                raise
            current = request.model_copy(
                update={
                    "user_content": (
                        f"{request.user_content}\n\n"
                        "The previous response failed validation.\n"
                        f"Validation error:\n{last_error}\n"
                        "Correct only the fields named in the validation error. "
                        "Return JSON that matches the schema. "
                        "Return only a compact JSON object. "
                        "Do not repeat the schema or the examples."
                    )
                }
            )

    raise ValueError(last_error)