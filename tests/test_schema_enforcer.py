"""
test_schema_enforcer.py — Tests for guardrails/schema_enforcer.py

Coverage:
  - Valid JSON parsing
  - Markdown-fenced JSON extraction
  - Pydantic validation success and failure
  - SchemaViolation error details
  - enforce_safe non-raising variant
  - Nested schemas
"""

import pytest
from pydantic import BaseModel
from typing import Literal
from guardrails.schema_enforcer import enforce, enforce_safe, SchemaViolation, _extract_json


# ── Test schemas ──────────────────────────────────────────────────────────

class SentimentSchema(BaseModel):
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    confidence: float

class EntitySchema(BaseModel):
    text: str
    entity_type: str

class EntityListSchema(BaseModel):
    entities: list[EntitySchema]


# ── JSON extraction ───────────────────────────────────────────────────────

class TestExtractJSON:
    def test_plain_json(self):
        raw = '{"key": "value"}'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        extracted = _extract_json(raw)
        assert '"key"' in extracted

    def test_with_leading_text(self):
        raw = 'Here is the result: {"sentiment": "POSITIVE", "confidence": 0.9}'
        extracted = _extract_json(raw)
        assert '"sentiment"' in extracted

    def test_array(self):
        raw = '[{"a": 1}, {"b": 2}]'
        extracted = _extract_json(raw)
        assert extracted.startswith("[")


# ── enforce() ─────────────────────────────────────────────────────────────

class TestEnforce:
    def test_valid_json(self):
        raw = '{"sentiment": "POSITIVE", "confidence": 0.92}'
        result = enforce(raw, SentimentSchema)
        assert result.sentiment == "POSITIVE"
        assert result.confidence == pytest.approx(0.92)

    def test_fenced_json(self):
        raw = '```json\n{"sentiment": "NEGATIVE", "confidence": 0.78}\n```'
        result = enforce(raw, SentimentSchema)
        assert result.sentiment == "NEGATIVE"

    def test_invalid_enum_raises(self):
        raw = '{"sentiment": "UNKNOWN", "confidence": 0.5}'
        with pytest.raises(SchemaViolation) as exc_info:
            enforce(raw, SentimentSchema)
        assert exc_info.value.model_name == "SentimentSchema"

    def test_missing_field_raises(self):
        raw = '{"sentiment": "POSITIVE"}'
        with pytest.raises(SchemaViolation):
            enforce(raw, SentimentSchema)

    def test_non_json_raises(self):
        with pytest.raises(SchemaViolation):
            enforce("This is not JSON at all.", SentimentSchema)

    def test_nested_schema(self):
        raw = '{"entities": [{"text": "JPMorgan", "entity_type": "ORG"}]}'
        result = enforce(raw, EntityListSchema)
        assert len(result.entities) == 1
        assert result.entities[0].text == "JPMorgan"

    def test_confidence_out_of_range_is_allowed(self):
        """Pydantic doesn't enforce range unless we add validators — just test parsing."""
        raw = '{"sentiment": "NEUTRAL", "confidence": 1.5}'
        result = enforce(raw, SentimentSchema)
        assert result.confidence == pytest.approx(1.5)


# ── enforce_safe() ────────────────────────────────────────────────────────

class TestEnforceSafe:
    def test_valid_returns_instance(self):
        raw = '{"sentiment": "POSITIVE", "confidence": 0.8}'
        instance, err = enforce_safe(raw, SentimentSchema)
        assert instance is not None
        assert err is None

    def test_invalid_returns_error(self):
        raw = "not json"
        instance, err = enforce_safe(raw, SentimentSchema)
        assert instance is None
        assert err is not None
        assert isinstance(err, SchemaViolation)

    def test_error_has_raw_text(self):
        raw = "not json"
        _, err = enforce_safe(raw, SentimentSchema)
        assert err.raw == raw

    def test_error_has_model_name(self):
        raw = '{"bad": "data"}'
        _, err = enforce_safe(raw, SentimentSchema)
        assert err.model_name == "SentimentSchema"
