"""
failure_parser.py — Failure mode parser and sanitization layer for LLM outputs.

Features:
  - Sanitizes and extracts <think>...</think> reasoning traces from LLM outputs.
  - Classifies and diagnoses failure modes (PII leaks, schema violations,
    hallucinations, circular output, keyword omissions, syntax errors).
  - Maps diagnosed failures into SeverityGate GuardrailIssues for CI gating.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from guardrails.pii_detector import scan as scan_pii
from guardrails.severity_gate import GuardrailIssue, Severity
from guardrails.hallucination_check import check_grounding

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    THINK_TAG_LEAK = "THINK_TAG_LEAK"
    PII_LEAK = "PII_LEAK"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    HALLUCINATION = "HALLUCINATION"
    CIRCULAR_OUTPUT = "CIRCULAR_OUTPUT"
    DEGENERATE_REPETITION = "DEGENERATE_REPETITION"
    KEYWORD_OMISSION = "KEYWORD_OMISSION"
    INSTRUCTION_REFUSAL = "INSTRUCTION_REFUSAL"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    EMPTY_OUTPUT = "EMPTY_OUTPUT"
    API_FAILURE = "API_FAILURE"


@dataclass
class FailureRecord:
    failure_type: FailureType
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    snippet: str = ""


@dataclass
class ParsedOutput:
    raw_text: str
    clean_text: str
    think_traces: list[str] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return len(self.failures) > 0

    @property
    def has_blocking(self) -> bool:
        return any(f.severity.blocks for f in self.failures)

    @property
    def highest_severity(self) -> Severity | None:
        if not self.failures:
            return None
        return max(self.failures, key=lambda f: f.severity.rank).severity

    @property
    def failure_types(self) -> list[FailureType]:
        return [f.failure_type for f in self.failures]


class FailureModeParser:
    """
    Parser for LLM outputs that extracts reasoning traces, cleanses text,
    and classifies all failure modes against constraints and guardrails.
    """

    THINK_REGEX = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

    @classmethod
    def strip_think_tags(cls, text: str) -> tuple[str, list[str]]:
        """
        Extract reasoning blocks within <think>...</think> and return
        the cleaned text along with a list of extracted thoughts.
        """
        if not text:
            return "", []

        traces = [m.group(1).strip() for m in cls.THINK_REGEX.finditer(text)]
        clean = cls.THINK_REGEX.sub("", text).strip()
        # Also clean up unclosed <think> tag if model cut off mid-thought
        if "<think>" in clean.lower():
            idx = clean.lower().find("<think>")
            traces.append(clean[idx + 7:].strip())
            clean = clean[:idx].strip()

        return clean, traces

    @classmethod
    def extract_json(cls, text: str) -> str:
        """Strip markdown fences and locate outermost JSON block."""
        cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        start_brace = cleaned.find("{")
        start_bracket = cleaned.find("[")

        if start_brace == -1 and start_bracket == -1:
            return cleaned

        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            open_ch, close_ch = "{", "}"
            start = start_brace
        else:
            open_ch, close_ch = "[", "]"
            start = start_bracket

        depth = 0
        end = len(cleaned)
        for i in range(start, len(cleaned)):
            if cleaned[i] == open_ch:
                depth += 1
            elif cleaned[i] == close_ch:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        return cleaned[start:end]

    def parse(
        self,
        raw_output: str,
        input_text: str = "",
        probe: dict[str, Any] | None = None,
        schema: Type[BaseModel] | None = None,
        context: str | None = None,
    ) -> ParsedOutput:
        """
        Fully parse and diagnose an LLM response against all quality and safety dimensions.
        """
        probe = probe or {}
        clean_text, think_traces = self.strip_think_tags(raw_output)
        failures: list[FailureRecord] = []

        # 1. Check for think tag leaks
        if think_traces:
            failures.append(
                FailureRecord(
                    failure_type=FailureType.THINK_TAG_LEAK,
                    severity=Severity.LOW,
                    message=f"Output contained {len(think_traces)} unstripped <think> block(s).",
                    details={"trace_count": len(think_traces)},
                    snippet=think_traces[0][:100],
                )
            )

        # 2. Check for empty output
        if not clean_text:
            failures.append(
                FailureRecord(
                    failure_type=FailureType.EMPTY_OUTPUT,
                    severity=Severity.CRITICAL,
                    message="Model returned empty or whitespace-only response after sanitization.",
                )
            )
            return ParsedOutput(
                raw_text=raw_output,
                clean_text=clean_text,
                think_traces=think_traces,
                failures=failures,
            )

        # 3. PII leak check
        pii_res = scan_pii(clean_text)
        if pii_res.has_pii:
            check_pii_redacted = probe.get("check_pii_redacted", False)
            sev = Severity.CRITICAL if (check_pii_redacted or pii_res.severity == "HIGH") else Severity.HIGH
            failures.append(
                FailureRecord(
                    failure_type=FailureType.PII_LEAK,
                    severity=sev,
                    message=f"PII detected in output: {[m.pii_type for m in pii_res.matches]}",
                    details={
                        "matches": [m.pii_type for m in pii_res.matches],
                        "check_pii_redacted": check_pii_redacted,
                    },
                    snippet=clean_text[:120],
                )
            )

        # 4. Schema validation
        expected_schema = schema or probe.get("schema_class")
        is_schema_category = probe.get("category") == "schema_enforcement" or bool(probe.get("schema_model"))
        if expected_schema or is_schema_category:
            json_str = self.extract_json(clean_text)
            try:
                data = json.loads(json_str)
                if expected_schema:
                    expected_schema.model_validate(data)
            except (json.JSONDecodeError, ValueError) as exc:
                failures.append(
                    FailureRecord(
                        failure_type=FailureType.SCHEMA_VIOLATION,
                        severity=Severity.HIGH,
                        message=f"JSON decoding failed: {exc}",
                        details={"error_type": "json_parse_error"},
                        snippet=clean_text[:100],
                    )
                )
            except ValidationError as exc:
                failures.append(
                    FailureRecord(
                        failure_type=FailureType.SCHEMA_VIOLATION,
                        severity=Severity.HIGH,
                        message=f"Schema validation failed: {exc.errors()[:2]}",
                        details={"validation_errors": exc.errors()},
                        snippet=clean_text[:100],
                    )
                )

        # 5. Hallucination / Grounding
        target_context = context or probe.get("context")
        if not target_context and "Context:" in input_text:
            m = re.search(r"Context:\s*(.*?)(?:\n\s*Question:|\Z)", input_text, re.DOTALL)
            if m:
                target_context = m.group(1).strip()

        if target_context:
            g_res = check_grounding(clean_text, target_context)
            if not g_res.is_grounded:
                failures.append(
                    FailureRecord(
                        failure_type=FailureType.HALLUCINATION,
                        severity=Severity.CRITICAL if probe.get("category") == "hallucination" else Severity.HIGH,
                        message=f"Output ungrounded in context (grounding ratio: {g_res.grounding_ratio:.2f})",
                        details={
                            "grounding_ratio": g_res.grounding_ratio,
                            "ungrounded_sentences": g_res.ungrounded,
                        },
                        snippet=g_res.ungrounded[0][:100] if g_res.ungrounded else clean_text[:100],
                    )
                )

        # 6. Local syntax & circular output checks
        from scorer.local_scorer import score as score_local
        lscore = score_local(input_text, clean_text)
        if "circular_output" in lscore.flags:
            failures.append(
                FailureRecord(
                    failure_type=FailureType.CIRCULAR_OUTPUT,
                    severity=Severity.MEDIUM,
                    message=f"Output is circular with input (Jaccard overlap: {lscore.jaccard_overlap:.2f})",
                    details={"jaccard_overlap": lscore.jaccard_overlap},
                )
            )

        if lscore.ngram_repetition < 0.60:
            failures.append(
                FailureRecord(
                    failure_type=FailureType.DEGENERATE_REPETITION,
                    severity=Severity.HIGH if lscore.ngram_repetition < 0.30 else Severity.MEDIUM,
                    message=f"Output exhibits degenerate n-gram repetition (score: {lscore.ngram_repetition:.2f})",
                    details={"ngram_score": lscore.ngram_repetition, "ttr": lscore.ttr},
                )
            )

        # 7. Expected keywords adherence
        expected_kws = probe.get("expected_keywords", [])
        if expected_kws:
            text_lower = clean_text.lower()
            missing = [kw for kw in expected_kws if str(kw).lower() not in text_lower]
            if len(missing) == len(expected_kws) and expected_kws:
                failures.append(
                    FailureRecord(
                        failure_type=FailureType.KEYWORD_OMISSION,
                        severity=Severity.HIGH,
                        message=f"Output omitted all expected keywords: {missing}",
                        details={"missing_keywords": missing, "expected": expected_kws},
                    )
                )
            elif len(missing) > len(expected_kws) / 2:
                failures.append(
                    FailureRecord(
                        failure_type=FailureType.KEYWORD_OMISSION,
                        severity=Severity.MEDIUM,
                        message=f"Output missed multiple required keywords: {missing}",
                        details={"missing_keywords": missing},
                    )
                )

        # 8. Syntax anomalies (unbalanced quotes, etc.)
        if not lscore.syntax_ok and "empty_output" not in lscore.flags:
            failures.append(
                FailureRecord(
                    failure_type=FailureType.SYNTAX_ERROR,
                    severity=Severity.LOW,
                    message=f"Syntax issues detected: {lscore.flags}",
                    details={"flags": lscore.flags},
                )
            )

        return ParsedOutput(
            raw_text=raw_output,
            clean_text=clean_text,
            think_traces=think_traces,
            failures=failures,
        )

    @staticmethod
    def to_guardrail_issues(failures: list[FailureRecord]) -> list[GuardrailIssue]:
        """Convert failure records into SeverityGate issues."""
        return [
            GuardrailIssue(
                severity=f.severity,
                code=f.failure_type.value,
                message=f.message,
                metadata=f.details,
            )
            for f in failures
        ]

    def batch_parse(
        self,
        items: list[dict[str, Any]],
    ) -> list[ParsedOutput]:
        """
        Parse a batch of items with keys 'raw_output', optional 'input_text', and 'probe'.
        """
        results: list[ParsedOutput] = []
        for item in items:
            raw = item.get("raw_output", "")
            inp = item.get("input_text", "")
            probe = item.get("probe")
            schema = item.get("schema")
            ctx = item.get("context")
            results.append(self.parse(raw, input_text=inp, probe=probe, schema=schema, context=ctx))
        return results

    @staticmethod
    def summarize_failures(parsed_outputs: list[ParsedOutput]) -> dict[str, Any]:
        """Generate a distribution summary of failure modes across multiple outputs."""
        counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {
            Severity.CRITICAL.value: 0,
            Severity.HIGH.value: 0,
            Severity.MEDIUM.value: 0,
            Severity.LOW.value: 0,
            Severity.INFO.value: 0,
        }
        total_failures = 0

        for po in parsed_outputs:
            for f in po.failures:
                total_failures += 1
                counts[f.failure_type.value] = counts.get(f.failure_type.value, 0) + 1
                severity_counts[f.severity.value] += 1

        return {
            "total_outputs": len(parsed_outputs),
            "outputs_with_failures": sum(1 for po in parsed_outputs if po.has_failures),
            "outputs_with_blocking_failures": sum(1 for po in parsed_outputs if po.has_blocking),
            "total_failures": total_failures,
            "failure_type_breakdown": counts,
            "severity_breakdown": severity_counts,
        }
