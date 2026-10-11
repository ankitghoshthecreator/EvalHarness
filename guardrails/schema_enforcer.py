"""
schema_enforcer.py — Pydantic schema enforcement for LLM outputs.

Validates that LLM-generated JSON matches the expected output schema.
Handles:
  - JSON extraction from raw LLM text (strips markdown fences)
  - Strict Pydantic v2 validation
  - Detailed error reporting
"""

from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class SchemaViolation(Exception):
    """Raised when LLM output doesn't conform to the expected schema."""

    def __init__(self, model_name: str, errors: list[dict], raw: str):
        self.model_name = model_name
        self.errors = errors
        self.raw = raw
        super().__init__(f"Schema violation for {model_name}: {errors}")


def _extract_json(text: str) -> str:
    """
    Extract JSON from LLM output that may be wrapped in markdown fences
    or have leading/trailing text.
    """
    # Strip ```json ... ``` fences
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    # Find first { or [ and last matching } or ]
    start = min(
        (text.find("{") if "{" in text else len(text)),
        (text.find("[") if "[" in text else len(text)),
    )
    if start == len(text):
        return text

    # Find matching closing bracket
    bracket = "{" if text[start] == "{" else "["
    close = "}" if bracket == "{" else "]"
    depth = 0
    end = start
    for i, ch in enumerate(text[start:], start=start):
        if ch == bracket:
            depth += 1
        elif ch == close:
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    return text[start:end]


def enforce(raw_text: str, schema: Type[T]) -> T:
    """
    Parse and validate LLM output against a Pydantic model.

    Args:
        raw_text: Raw LLM response string.
        schema: Pydantic model class to validate against.

    Returns:
        Validated instance of `schema`.

    Raises:
        SchemaViolation: If parsing or validation fails.
    """
    try:
        json_str = _extract_json(raw_text)
        data: Any = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SchemaViolation(
            model_name=schema.__name__,
            errors=[{"type": "json_parse_error", "msg": str(exc)}],
            raw=raw_text,
        ) from exc

    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise SchemaViolation(
            model_name=schema.__name__,
            errors=exc.errors(),
            raw=raw_text,
        ) from exc


def enforce_safe(raw_text: str, schema: Type[T]) -> tuple[T | None, SchemaViolation | None]:
    """
    Non-raising version of enforce(). Returns (instance, None) or (None, error).
    """
    try:
        return enforce(raw_text, schema), None
    except SchemaViolation as exc:
        return None, exc
