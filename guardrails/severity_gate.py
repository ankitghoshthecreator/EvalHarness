"""
severity_gate.py — Severity-gated guardrail enforcement.

Features:
  - CRITICAL / HIGH  → block pipeline (raise SeverityBlock)
  - MEDIUM / LOW     → log warning, continue
  - Priority hard-capped at 5 issues per run

Severity levels: CRITICAL > HIGH > MEDIUM > LOW > INFO
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)

MAX_ISSUES = 5  # Hard cap for blocking pipeline issues per run


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

    @property
    def blocks(self) -> bool:
        return self in (Severity.CRITICAL, Severity.HIGH)

    @property
    def rank(self) -> int:
        return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}[self.value]


@dataclass
class GuardrailIssue:
    severity: Severity
    code: str          # machine-readable identifier e.g. "PII_DETECTED"
    message: str
    metadata: dict = field(default_factory=dict)


class SeverityBlock(Exception):
    """Raised when CRITICAL or HIGH issues are detected and must block the pipeline."""

    def __init__(self, issues: list[GuardrailIssue]):
        self.issues = issues
        codes = [i.code for i in issues]
        super().__init__(f"Pipeline blocked by {len(issues)} issue(s): {codes}")


class SeverityGate:
    """
    Collects issues from multiple guardrail checks and enforces severity gating.

    Usage:
        gate = SeverityGate()
        gate.add(GuardrailIssue(Severity.HIGH, "PII_DETECTED", "SSN found in output"))
        gate.enforce()   # raises SeverityBlock if any CRITICAL/HIGH present
    """

    def __init__(self, max_issues: int = MAX_ISSUES):
        self._issues: list[GuardrailIssue] = []
        self._max = max_issues

    def add(self, issue: GuardrailIssue) -> None:
        if len(self._issues) >= self._max:
            logger.warning("SeverityGate: max_issues cap (%d) reached; ignoring further issues.", self._max)
            return
        self._issues.append(issue)
        if issue.severity.blocks:
            logger.error("[%s] %s: %s", issue.severity.value, issue.code, issue.message)
        else:
            logger.warning("[%s] %s: %s", issue.severity.value, issue.code, issue.message)

    def add_if(
        self,
        condition: bool,
        severity: Severity,
        code: str,
        message: str,
        metadata: dict | None = None,
    ) -> None:
        """Convenience: only add issue if condition is True."""
        if condition:
            self.add(GuardrailIssue(severity, code, message, metadata or {}))

    @property
    def issues(self) -> list[GuardrailIssue]:
        return list(self._issues)

    @property
    def blocking_issues(self) -> list[GuardrailIssue]:
        return [i for i in self._issues if i.severity.blocks]

    @property
    def has_blocking(self) -> bool:
        return bool(self.blocking_issues)

    def enforce(self) -> None:
        """
        Raise SeverityBlock if any CRITICAL or HIGH issues exist.
        LOW/MEDIUM issues are already logged but do not block.
        """
        if self.has_blocking:
            raise SeverityBlock(self.blocking_issues)

    def summary(self) -> dict:
        return {
            "total_issues": len(self._issues),
            "blocking": len(self.blocking_issues),
            "issues": [
                {
                    "severity": i.severity.value,
                    "code": i.code,
                    "message": i.message,
                }
                for i in sorted(self._issues, key=lambda x: -x.severity.rank)
            ],
        }
