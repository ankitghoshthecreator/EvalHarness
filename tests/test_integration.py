"""
test_severity_gate.py — Tests for guardrails/severity_gate.py
test_report_generator.py — Tests for reporter/report_generator.py
test_ragas_metrics.py — Tests for scorer/ragas_metrics.py
test_runner.py — Integration tests for runner.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# Severity Gate Tests
# ═══════════════════════════════════════════════════════════════════════════

import pytest
from guardrails.severity_gate import (
    Severity, SeverityGate, SeverityBlock, GuardrailIssue
)


class TestSeverityEnum:
    def test_critical_blocks(self):
        assert Severity.CRITICAL.blocks is True

    def test_high_blocks(self):
        assert Severity.HIGH.blocks is True

    def test_medium_not_blocks(self):
        assert Severity.MEDIUM.blocks is False

    def test_low_not_blocks(self):
        assert Severity.LOW.blocks is False

    def test_rank_ordering(self):
        assert Severity.CRITICAL.rank > Severity.HIGH.rank > Severity.MEDIUM.rank > Severity.LOW.rank


class TestSeverityGate:
    def test_clean_gate_doesnt_raise(self):
        gate = SeverityGate()
        gate.enforce()  # should not raise

    def test_critical_raises(self):
        gate = SeverityGate()
        gate.add(GuardrailIssue(Severity.CRITICAL, "TEST", "Critical issue"))
        with pytest.raises(SeverityBlock):
            gate.enforce()

    def test_high_raises(self):
        gate = SeverityGate()
        gate.add(GuardrailIssue(Severity.HIGH, "TEST_HIGH", "High severity"))
        with pytest.raises(SeverityBlock):
            gate.enforce()

    def test_medium_doesnt_raise(self):
        gate = SeverityGate()
        gate.add(GuardrailIssue(Severity.MEDIUM, "MED", "Medium issue"))
        gate.enforce()  # should NOT raise

    def test_low_doesnt_raise(self):
        gate = SeverityGate()
        gate.add(GuardrailIssue(Severity.LOW, "LOW", "Low issue"))
        gate.enforce()  # should NOT raise

    def test_max_issues_cap(self):
        gate = SeverityGate(max_issues=3)
        for i in range(10):
            gate.add(GuardrailIssue(Severity.LOW, f"CODE_{i}", "msg"))
        assert len(gate.issues) == 3

    def test_add_if_true(self):
        gate = SeverityGate()
        gate.add_if(True, Severity.HIGH, "CODE", "message")
        assert len(gate.issues) == 1

    def test_add_if_false(self):
        gate = SeverityGate()
        gate.add_if(False, Severity.HIGH, "CODE", "message")
        assert len(gate.issues) == 0

    def test_blocking_issues_filter(self):
        gate = SeverityGate()
        gate.add(GuardrailIssue(Severity.HIGH, "H", "high"))
        gate.add(GuardrailIssue(Severity.LOW, "L", "low"))
        assert len(gate.blocking_issues) == 1

    def test_summary_structure(self):
        gate = SeverityGate()
        gate.add(GuardrailIssue(Severity.HIGH, "H", "msg"))
        summary = gate.summary()
        assert "total_issues" in summary
        assert "blocking" in summary
        assert "issues" in summary

    def test_severity_block_contains_issues(self):
        gate = SeverityGate()
        gate.add(GuardrailIssue(Severity.CRITICAL, "CRIT", "msg"))
        try:
            gate.enforce()
        except SeverityBlock as e:
            assert len(e.issues) == 1
            assert e.issues[0].code == "CRIT"


# ═══════════════════════════════════════════════════════════════════════════
# Report Generator Tests
# ═══════════════════════════════════════════════════════════════════════════

from reporter.report_generator import (
    ProbeResult, EvalRunReport, render_markdown, render_json,
    regression_diff, render_regression_markdown, save_report,
)
import json
import tempfile
from pathlib import Path


def _make_result(pid: str, composite: float, passed: bool = None) -> ProbeResult:
    return ProbeResult(
        probe_id=pid,
        category="test",
        local_score=composite,
        judge_score=None,
        composite=composite,
        passed=(composite >= 0.60) if passed is None else passed,
    )


class TestEvalRunReport:
    def test_build(self):
        results = [_make_result("p1", 0.8), _make_result("p2", 0.4)]
        report = EvalRunReport.build(results, model="mock")
        assert report.total_probes == 2
        assert report.passed == 1
        assert report.failed == 1

    def test_pass_rate(self):
        results = [_make_result(f"p{i}", 0.9) for i in range(4)]
        report = EvalRunReport.build(results)
        assert report.pass_rate == 1.0

    def test_empty_results(self):
        report = EvalRunReport.build([])
        assert report.total_probes == 0
        assert report.pass_rate == 0.0


class TestRenderMarkdown:
    def test_contains_run_id(self):
        results = [_make_result("p1", 0.8)]
        report = EvalRunReport.build(results)
        md = render_markdown(report)
        assert report.run_id in md

    def test_contains_probe_id(self):
        results = [_make_result("sum_001", 0.8)]
        report = EvalRunReport.build(results)
        md = render_markdown(report)
        assert "sum_001" in md


class TestRenderJSON:
    def test_valid_json(self):
        results = [_make_result("p1", 0.8)]
        report = EvalRunReport.build(results)
        parsed = json.loads(render_json(report))
        assert parsed["total_probes"] == 1

    def test_results_in_json(self):
        results = [_make_result("myprobe", 0.75)]
        report = EvalRunReport.build(results)
        parsed = json.loads(render_json(report))
        assert any(r["probe_id"] == "myprobe" for r in parsed["results"])


class TestRegressionDiff:
    def test_stable(self):
        r1 = [_make_result("p1", 0.8)]
        r2 = [_make_result("p1", 0.81)]
        baseline = EvalRunReport.build(r1)
        current = EvalRunReport.build(r2)
        entries = regression_diff(baseline, current)
        assert entries[0].status == "STABLE"

    def test_regression_detected(self):
        r1 = [_make_result("p1", 0.9)]
        r2 = [_make_result("p1", 0.5)]
        baseline = EvalRunReport.build(r1)
        current = EvalRunReport.build(r2)
        entries = regression_diff(baseline, current)
        assert entries[0].status == "REGRESSED"

    def test_improvement_detected(self):
        r1 = [_make_result("p1", 0.5)]
        r2 = [_make_result("p1", 0.9)]
        baseline = EvalRunReport.build(r1)
        current = EvalRunReport.build(r2)
        entries = regression_diff(baseline, current)
        assert entries[0].status == "IMPROVED"

    def test_new_probe(self):
        baseline = EvalRunReport.build([_make_result("p1", 0.8)])
        current = EvalRunReport.build([_make_result("p1", 0.8), _make_result("p2", 0.7)])
        entries = regression_diff(baseline, current)
        new_entry = next(e for e in entries if e.probe_id == "p2")
        assert new_entry.status == "NEW"


class TestSaveReport:
    def test_creates_files(self):
        results = [_make_result("p1", 0.8)]
        report = EvalRunReport.build(results, run_id="test_run_001")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_report(report, tmpdir)
            assert paths["json"].exists()
            assert paths["markdown"].exists()


# ═══════════════════════════════════════════════════════════════════════════
# RAGAS Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════

from scorer.ragas_metrics import (
    faithfulness, context_relevancy, recall_at_k,
    precision_at_1, mean_reciprocal_rank, evaluate_rag, RAGASResult,
)


class TestFaithfulness:
    def test_fully_grounded(self):
        context = "Apple revenue was 90 billion dollars in Q4 2024."
        output = "Apple revenue was 90 billion dollars."
        score = faithfulness(output, context)
        assert score > 0.5

    def test_ungrounded(self):
        context = "Apple revenue was 90 billion dollars."
        output = "Microsoft announced quantum computing breakthroughs."
        score = faithfulness(output, context)
        assert score < 0.5

    def test_empty_output(self):
        assert faithfulness("", "some context") == 0.0


class TestContextRelevancy:
    def test_relevant_context(self):
        query = "What is the interest rate?"
        context = "The Federal Reserve raised interest rates to 5.5%."
        score = context_relevancy(query, context)
        assert score > 0.0

    def test_irrelevant_context(self):
        query = "What is the interest rate?"
        context = "The cat sat on the mat and purred loudly."
        score = context_relevancy(query, context)
        assert score <= 0.3


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["d1", "d2", "d3"], {"d1", "d2", "d3"}, k=5) == 1.0

    def test_none_relevant(self):
        assert recall_at_k(["d1", "d2"], {"d3", "d4"}, k=5) == 0.0

    def test_partial(self):
        score = recall_at_k(["d1", "d2", "d3"], {"d1", "d4"}, k=5)
        assert score == pytest.approx(0.5)

    def test_k_limits(self):
        # With k=1, only d1 is checked
        score = recall_at_k(["d1", "d2"], {"d2"}, k=1)
        assert score == 0.0


class TestPrecisionAt1:
    def test_first_is_relevant(self):
        assert precision_at_1(["d1", "d2"], {"d1"}) == 1.0

    def test_first_not_relevant(self):
        assert precision_at_1(["d2", "d1"], {"d1"}) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_1([], {"d1"}) == 0.0


class TestMRR:
    def test_first_relevant(self):
        assert mean_reciprocal_rank(["d1", "d2"], {"d1"}) == 1.0

    def test_second_relevant(self):
        assert mean_reciprocal_rank(["d2", "d1"], {"d1"}) == pytest.approx(0.5)

    def test_no_relevant(self):
        assert mean_reciprocal_rank(["d2", "d3"], {"d1"}) == 0.0


class TestEvaluateRAG:
    def test_returns_ragas_result(self):
        result = evaluate_rag(
            query="What is the rate?",
            output_text="The rate is 5.5%.",
            context="The Federal Reserve set the rate to 5.5%.",
            retrieved_ids=["doc1"],
            relevant_ids={"doc1"},
        )
        assert isinstance(result, RAGASResult)
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.recall_at_k <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Runner (mock LLM)
# ═══════════════════════════════════════════════════════════════════════════

from runner import run_harness, load_probes, evaluate_probe, _mock_llm


class TestRunnerIntegration:
    def test_load_probes(self, probe_bank_path):
        probes = load_probes(probe_bank_path)
        assert len(probes) > 0
        assert all("id" in p for p in probes)
        assert all("input" in p for p in probes)

    def test_evaluate_probe_mock(self, probe_bank_path):
        probes = load_probes(probe_bank_path)
        probe = probes[0]
        result = evaluate_probe(probe, _mock_llm, judge=None)
        assert result.probe_id == probe["id"]
        assert 0.0 <= result.composite <= 1.0
        assert isinstance(result.passed, bool)

    def test_full_harness_mock(self, probe_bank_path, tmp_path):
        report = run_harness(
            probe_bank_path=str(probe_bank_path),
            model="mock",
            output_dir=str(tmp_path),
        )
        assert report.total_probes > 0
        assert 0.0 <= report.pass_rate <= 1.0

    def test_mock_llm_returns_string(self):
        result = _mock_llm("Summarize this text.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_llm_json_prompt(self):
        result = _mock_llm("Return a JSON object with key and value.")
        assert "{" in result


# ═══════════════════════════════════════════════════════════════════════════
# Hallucination Checker Tests
# ═══════════════════════════════════════════════════════════════════════════

from guardrails.hallucination_check import check_grounding, GroundingResult


class TestHallucinationCheck:
    def test_grounded_text(self):
        context = "JPMorgan Chase reported revenue of $38.5 billion for Q1 2024 with record net income."
        output = "JPMorgan Chase reported revenue of $38.5 billion for Q1 2024."
        res = check_grounding(output, context)
        assert isinstance(res, GroundingResult)
        assert res.is_grounded is True
        assert res.grounding_ratio > 0.6
        assert len(res.ungrounded) == 0

    def test_ungrounded_text(self):
        context = "JPMorgan Chase reported revenue of $38.5 billion for Q1 2024."
        output = "Tesla launched a brand new electric semi truck across European markets."
        res = check_grounding(output, context)
        assert res.is_grounded is False
        assert len(res.ungrounded) > 0

    def test_empty_output(self):
        res = check_grounding("", "Some context here.")
        assert res.is_grounded is True
        assert res.grounding_ratio == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# A/B Comparator Tests
# ═══════════════════════════════════════════════════════════════════════════

from reporter.ab_comparator import compare, render_ab_markdown, ABReport, ProbeFlip


class TestABComparator:
    def test_compare_winner_b(self):
        r1 = [_make_result("p1", 0.5, passed=False), _make_result("p2", 0.6, passed=True)]
        r2 = [_make_result("p1", 0.9, passed=True), _make_result("p2", 0.85, passed=True)]
        rep_a = EvalRunReport.build(r1, run_id="v1")
        rep_b = EvalRunReport.build(r2, run_id="v2")

        ab = compare(rep_a, rep_b, version_a="Prompt_A", version_b="Prompt_B")
        assert isinstance(ab, ABReport)
        assert ab.winner == "B"
        assert ab.avg_delta > 0
        assert any(f.flip == "FAIL→PASS" for f in ab.flips)

    def test_compare_winner_a(self):
        r1 = [_make_result("p1", 0.9, passed=True)]
        r2 = [_make_result("p1", 0.4, passed=False)]
        rep_a = EvalRunReport.build(r1)
        rep_b = EvalRunReport.build(r2)

        ab = compare(rep_a, rep_b)
        assert ab.winner == "A"
        assert ab.avg_delta < 0
        assert ab.flips[0].flip == "PASS→FAIL"

    def test_compare_tie(self):
        r1 = [_make_result("p1", 0.8, passed=True)]
        r2 = [_make_result("p1", 0.8, passed=True)]
        rep_a = EvalRunReport.build(r1)
        rep_b = EvalRunReport.build(r2)

        ab = compare(rep_a, rep_b)
        assert ab.winner == "TIE"

    def test_summary_and_markdown(self):
        r1 = [_make_result("p1", 0.4, passed=False)]
        r2 = [_make_result("p1", 0.9, passed=True)]
        ab = compare(EvalRunReport.build(r1), EvalRunReport.build(r2))

        summary = ab.summary()
        assert "A/B Comparison" in summary
        assert "Winner" in summary

        md = render_ab_markdown(ab)
        assert "# A/B Prompt Comparison" in md
        assert "p1" in md
        assert "FAIL→PASS" in md


# ═══════════════════════════════════════════════════════════════════════════
# Judge Agent Unit Tests (offline / mocked)
# ═══════════════════════════════════════════════════════════════════════════

from scorer.judge_agent import JudgeAgent, JudgeReport, _strip_think_tags
from unittest.mock import patch


class TestJudgeAgent:
    def test_strip_think_tags(self):
        text = "<think>Let me ponder the question carefully.</think>{\"fluency\": 0.9}"
        assert _strip_think_tags(text) == '{"fluency": 0.9}'

    def test_judge_report_composite_calculation(self):
        rep = JudgeReport(
            fluency=0.9,
            factual_grounding=0.9,
            instruction_adherence=0.9,
            hallucination_risk="NONE",
            rationale="Excellent response",
        )
        assert rep.composite == pytest.approx(0.9, abs=0.01)

    def test_judge_report_penalty(self):
        rep_low = JudgeReport(
            fluency=0.9,
            factual_grounding=0.9,
            instruction_adherence=0.9,
            hallucination_risk="LOW",
        )
        assert rep_low.composite == pytest.approx(0.8, abs=0.01)

        rep_high = JudgeReport(
            fluency=0.9,
            factual_grounding=0.9,
            instruction_adherence=0.9,
            hallucination_risk="HIGH",
        )
        assert rep_high.composite == pytest.approx(0.5, abs=0.01)

    def test_judge_agent_fallback(self):
        agent = JudgeAgent(api_key="fake_key")
        fallback = agent._fallback(0.75)
        assert fallback.escalation_failed is True
        assert fallback.fluency == 0.75
        assert fallback.hallucination_risk == "LOW"

    def test_judge_agent_mock_evaluate(self):
        agent = JudgeAgent(api_key="fake_key")
        mock_json = '{"fluency": 0.85, "factual_grounding": 0.9, "instruction_adherence": 0.95, "hallucination_risk": "NONE", "rationale": "Clear and factual"}'
        with patch.object(agent, "_call_groq", return_value=f"<think>test</think>{mock_json}"):
            res = agent.evaluate("prompt", "response")
            assert res.fluency == 0.85
            assert res.factual_grounding == 0.9
            assert res.hallucination_risk == "NONE"
            assert res.escalation_failed is False
