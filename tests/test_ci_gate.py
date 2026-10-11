"""
test_ci_gate.py — Unit tests for CI/CD Quality Gate & Merge Blocker.
"""

import pytest
from ci_gate import GatePolicy, GateResult, evaluate_gate
from reporter.report_generator import EvalRunReport, ProbeResult


def _make_probe(pid: str, score: float, passed: bool, severity: str = "HIGH", cat: str = "general") -> ProbeResult:
    return ProbeResult(
        probe_id=pid,
        category=cat,
        local_score=score,
        judge_score=None,
        composite=score,
        passed=passed,
        severity=severity,
    )


class TestGateEvaluation:
    def test_gate_passes_clean_run(self):
        probes = [_make_probe(f"p_{i}", 0.95, passed=True) for i in range(10)]
        cand = EvalRunReport.build(probes)
        result = evaluate_gate(cand, baseline=None, policy=GatePolicy(min_pass_rate=0.90))

        assert result.passed is True
        assert len(result.blocking_reasons) == 0
        assert result.pass_rate == 1.0

    def test_gate_blocks_on_low_pass_rate(self):
        # 8 pass, 2 fail -> 80% pass rate
        probes = [_make_probe(f"p_{i}", 0.90, passed=True) for i in range(8)]
        probes.extend([_make_probe(f"f_{i}", 0.40, passed=False) for i in range(2)])
        cand = EvalRunReport.build(probes)

        result = evaluate_gate(cand, baseline=None, policy=GatePolicy(min_pass_rate=0.90))
        assert result.passed is False
        assert any("below minimum threshold" in r for r in result.blocking_reasons)

    def test_gate_blocks_on_critical_failure(self):
        probes = [_make_probe("p1", 0.95, passed=True)]
        probes.append(_make_probe("crit_1", 0.20, passed=False, severity="CRITICAL"))
        cand = EvalRunReport.build(probes)

        result = evaluate_gate(cand, baseline=None)
        assert result.passed is False
        assert any("CRITICAL" in r for r in result.blocking_reasons)

    def test_gate_blocks_on_critical_regression(self):
        # Baseline had crit_1 at 0.95 PASS
        base = EvalRunReport.build([_make_probe("crit_1", 0.95, passed=True, severity="CRITICAL")])
        # Candidate dropped to 0.60
        cand = EvalRunReport.build([_make_probe("crit_1", 0.60, passed=True, severity="CRITICAL")])

        result = evaluate_gate(cand, baseline=base, policy=GatePolicy(min_pass_rate=0.50))
        assert result.passed is False
        assert any("regressed" in r for r in result.blocking_reasons)

    def test_gate_allows_minor_warning(self):
        base = EvalRunReport.build([_make_probe("p1", 0.95, passed=True, severity="LOW")])
        cand = EvalRunReport.build([_make_probe("p1", 0.85, passed=True, severity="LOW")])

        # Policy allows up to 1 regression
        result = evaluate_gate(cand, baseline=base, policy=GatePolicy(min_pass_rate=0.50, max_regressed_probes=1))
        assert result.passed is True
        assert len(result.warnings) == 1
        assert "regressed slightly" in result.warnings[0]


class TestGateMarkdownRendering:
    def test_render_contains_key_elements(self):
        probes = [_make_probe("p1", 0.95, passed=True)]
        cand = EvalRunReport.build(probes)
        result = evaluate_gate(cand, baseline=None)
        md = result.render_markdown()

        assert "# CI/CD LLM Evaluation Quality Gate" in md
        assert "GATE PASSED" in md
        assert "Overall Pass Rate" in md
        assert "Target" in md
