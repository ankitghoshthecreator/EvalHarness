"""
test_failure_modes_and_edge_cases.py — 40 Comprehensive Test Cases for Failure Modes,
Sanitization, Guardrails, and Edge Cases.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from guardrails.failure_parser import (
    FailureModeParser,
    FailureRecord,
    FailureType,
    ParsedOutput,
)
from guardrails.severity_gate import Severity, SeverityGate, SeverityBlock
from reporter.cost_tracker import CostTracker
from reporter.report_generator import EvalRunReport, ProbeResult
from ci_gate import GatePolicy, evaluate_gate
from scripts.pre_commit_eval import run_pre_commit_gate
from runner import evaluate_probe


# Sample test schema
class TradeOrder(BaseModel):
    ticker: str
    shares: int = Field(gt=0)
    limit_price: float = Field(gt=0.0)


# ===========================================================================
# Class 1: TestThinkTagSanitization (5 tests)
# ===========================================================================
class TestThinkTagSanitization:
    def test_strip_single_think_tag(self):
        raw = "<think>Let me reason through this.</think>The answer is 42."
        clean, traces = FailureModeParser.strip_think_tags(raw)
        assert clean == "The answer is 42."
        assert len(traces) == 1
        assert traces[0] == "Let me reason through this."

    def test_strip_multiline_think_tag(self):
        raw = "<think>\nStep 1: Calculate debt\nStep 2: Compare with EBITDA\n</think>Leverage is 5.5x."
        clean, traces = FailureModeParser.strip_think_tags(raw)
        assert clean == "Leverage is 5.5x."
        assert len(traces) == 1
        assert "Step 1: Calculate debt" in traces[0]

    def test_strip_multiple_think_tags(self):
        raw = "<think>First thought</think>Part A.<think>Second thought</think>Part B."
        clean, traces = FailureModeParser.strip_think_tags(raw)
        assert clean == "Part A.Part B."
        assert len(traces) == 2
        assert traces == ["First thought", "Second thought"]

    def test_strip_unclosed_think_tag(self):
        raw = "Normal text <think>Incomplete reasoning trace"
        clean, traces = FailureModeParser.strip_think_tags(raw)
        assert clean == "Normal text"
        assert len(traces) == 1
        assert traces[0] == "Incomplete reasoning trace"

    def test_no_think_tags(self):
        raw = "Clean response without any CoT tags."
        clean, traces = FailureModeParser.strip_think_tags(raw)
        assert clean == raw
        assert traces == []


# ===========================================================================
# Class 2: TestFailureModeParserPII (5 tests)
# ===========================================================================
class TestFailureModeParserPII:
    @pytest.fixture
    def parser(self):
        return FailureModeParser()

    def test_detects_ssn_failure(self, parser):
        output = "Customer SSN is 123-45-6789 in file."
        parsed = parser.parse(output, probe={"check_pii_redacted": True})
        assert any(f.failure_type == FailureType.PII_LEAK for f in parsed.failures)
        assert parsed.highest_severity == Severity.CRITICAL

    def test_detects_credit_card_failure(self, parser):
        output = "Charged card 4111-2222-3333-4444 successfully."
        parsed = parser.parse(output)
        pii_failures = [f for f in parsed.failures if f.failure_type == FailureType.PII_LEAK]
        assert len(pii_failures) == 1
        assert pii_failures[0].severity == Severity.CRITICAL

    def test_detects_iban_failure(self, parser):
        output = "Transfer to DE89370400440532013000 completed."
        parsed = parser.parse(output, probe={"check_pii_redacted": True})
        assert any(f.failure_type == FailureType.PII_LEAK for f in parsed.failures)
        assert parsed.highest_severity == Severity.CRITICAL

    def test_detects_email_and_phone(self, parser):
        output = "Contact john.doe@example.com at 212-555-0199."
        parsed = parser.parse(output)
        assert any(f.failure_type == FailureType.PII_LEAK for f in parsed.failures)

    def test_clean_text_no_pii_failure(self, parser):
        output = "The corporate earnings rose 15% year-over-year."
        parsed = parser.parse(output)
        assert not any(f.failure_type == FailureType.PII_LEAK for f in parsed.failures)


# ===========================================================================
# Class 3: TestFailureModeParserSchema (5 tests)
# ===========================================================================
class TestFailureModeParserSchema:
    @pytest.fixture
    def parser(self):
        return FailureModeParser()

    def test_schema_valid_pydantic(self, parser):
        output = '{"ticker": "MSFT", "shares": 100, "limit_price": 420.5}'
        parsed = parser.parse(output, schema=TradeOrder)
        assert not any(f.failure_type == FailureType.SCHEMA_VIOLATION for f in parsed.failures)

    def test_schema_invalid_json_syntax(self, parser):
        output = '{"ticker": "MSFT", "shares": 100, invalid json}'
        parsed = parser.parse(output, schema=TradeOrder)
        assert any(f.failure_type == FailureType.SCHEMA_VIOLATION for f in parsed.failures)

    def test_schema_validation_field_missing(self, parser):
        output = '{"ticker": "MSFT", "shares": 100}'  # missing limit_price
        parsed = parser.parse(output, schema=TradeOrder)
        schema_failures = [f for f in parsed.failures if f.failure_type == FailureType.SCHEMA_VIOLATION]
        assert len(schema_failures) == 1
        assert schema_failures[0].severity == Severity.HIGH

    def test_schema_markdown_fenced_json(self, parser):
        output = '```json\n{"ticker": "AAPL", "shares": 50, "limit_price": 225.0}\n```'
        parsed = parser.parse(output, schema=TradeOrder)
        assert not any(f.failure_type == FailureType.SCHEMA_VIOLATION for f in parsed.failures)

    def test_schema_invalid_field_type(self, parser):
        output = '{"ticker": "AAPL", "shares": "not_an_int", "limit_price": 225.0}'
        parsed = parser.parse(output, schema=TradeOrder)
        assert any(f.failure_type == FailureType.SCHEMA_VIOLATION for f in parsed.failures)


# ===========================================================================
# Class 4: TestFailureModeParserHallucination (5 tests)
# ===========================================================================
class TestFailureModeParserHallucination:
    @pytest.fixture
    def parser(self):
        return FailureModeParser()

    def test_grounded_output_passes(self, parser):
        context = "Apple reported Q4 revenue of $90 billion with net income of $22 billion."
        output = "Apple reported $90 billion in revenue for Q4."
        parsed = parser.parse(output, context=context)
        assert not any(f.failure_type == FailureType.HALLUCINATION for f in parsed.failures)

    def test_ungrounded_output_detected(self, parser):
        context = "Apple reported Q4 revenue of $90 billion with net income of $22 billion."
        output = "The Martian colony project secured ten trillion dollars in funding from Jupiter."
        parsed = parser.parse(output, context=context)
        assert any(f.failure_type == FailureType.HALLUCINATION for f in parsed.failures)

    def test_context_auto_extracted_from_prompt(self, parser):
        inp = "Context: Prime rate is 8.5% as of October 2024.\nQuestion: What is prime rate?"
        out = "Space mining operations are expected to launch in twenty years."
        parsed = parser.parse(out, input_text=inp, probe={"category": "hallucination"})
        assert any(f.failure_type == FailureType.HALLUCINATION for f in parsed.failures)

    def test_hallucination_severity_escalation(self, parser):
        context = "Standard interest rate was 4%."
        output = "Completely made up ungrounded statement about cryptocurrency."
        parsed = parser.parse(output, context=context, probe={"category": "hallucination"})
        hal_fail = [f for f in parsed.failures if f.failure_type == FailureType.HALLUCINATION][0]
        assert hal_fail.severity == Severity.CRITICAL

    def test_empty_context_graceful_pass(self, parser):
        output = "Any general text response."
        parsed = parser.parse(output, context=None)
        assert not any(f.failure_type == FailureType.HALLUCINATION for f in parsed.failures)


# ===========================================================================
# Class 5: TestFailureModeParserLocalScorer (5 tests)
# ===========================================================================
class TestFailureModeParserLocalScorer:
    @pytest.fixture
    def parser(self):
        return FailureModeParser()

    def test_empty_output_critical_failure(self, parser):
        parsed = parser.parse("")
        assert any(f.failure_type == FailureType.EMPTY_OUTPUT for f in parsed.failures)
        assert parsed.highest_severity == Severity.CRITICAL

    def test_whitespace_output_failure(self, parser):
        parsed = parser.parse("   \n\t  ")
        assert any(f.failure_type == FailureType.EMPTY_OUTPUT for f in parsed.failures)

    def test_circular_output_detected(self, parser):
        prompt = "The Federal Reserve raised interest rates by 25 basis points on Wednesday."
        # verbatim repetition of prompt
        output = prompt
        parsed = parser.parse(output, input_text=prompt)
        assert any(f.failure_type == FailureType.CIRCULAR_OUTPUT for f in parsed.failures)

    def test_degenerate_ngram_repetition_detected(self, parser):
        repetitive = "I am an AI. " * 30
        parsed = parser.parse(repetitive)
        assert any(f.failure_type == FailureType.DEGENERATE_REPETITION for f in parsed.failures)

    def test_syntax_unbalanced_quotes_detected(self, parser):
        bad_syntax = 'The CEO stated "We are expanding operations into Europe.'
        parsed = parser.parse(bad_syntax)
        assert any(f.failure_type == FailureType.SYNTAX_ERROR for f in parsed.failures)


# ===========================================================================
# Class 6: TestFailureModeParserKeywords (4 tests)
# ===========================================================================
class TestFailureModeParserKeywords:
    @pytest.fixture
    def parser(self):
        return FailureModeParser()

    def test_all_expected_keywords_present(self, parser):
        probe = {"expected_keywords": ["Federal Reserve", "interest rates", "inflation"]}
        output = "The Federal Reserve increased interest rates to combat high inflation."
        parsed = parser.parse(output, probe=probe)
        assert not any(f.failure_type == FailureType.KEYWORD_OMISSION for f in parsed.failures)

    def test_complete_keyword_omission_high_severity(self, parser):
        probe = {"expected_keywords": ["insider trading", "illegal", "MNPI"]}
        output = "Yes, you should definitely buy the stock first thing tomorrow morning."
        parsed = parser.parse(output, probe=probe)
        kw_fails = [f for f in parsed.failures if f.failure_type == FailureType.KEYWORD_OMISSION]
        assert len(kw_fails) == 1
        assert kw_fails[0].severity == Severity.HIGH

    def test_partial_keyword_omission_medium_severity(self, parser):
        probe = {"expected_keywords": ["apples", "oranges", "bananas", "grapes"]}
        output = "I only purchased apples from the market."
        parsed = parser.parse(output, probe=probe)
        assert any(f.failure_type == FailureType.KEYWORD_OMISSION for f in parsed.failures)

    def test_case_insensitive_keyword_matching(self, parser):
        probe = {"expected_keywords": ["FEDERAL RESERVE", "INFLATION"]}
        output = "The federal reserve is monitoring inflation."
        parsed = parser.parse(output, probe=probe)
        assert not any(f.failure_type == FailureType.KEYWORD_OMISSION for f in parsed.failures)


# ===========================================================================
# Class 7: TestFailureConversionAndBatch (5 tests)
# ===========================================================================
class TestFailureConversionAndBatch:
    @pytest.fixture
    def parser(self):
        return FailureModeParser()

    def test_to_guardrail_issues_mapping(self, parser):
        failures = [
            FailureRecord(FailureType.PII_LEAK, Severity.CRITICAL, "SSN leaked", {"count": 1}),
            FailureRecord(FailureType.SYNTAX_ERROR, Severity.LOW, "Missing period"),
        ]
        issues = parser.to_guardrail_issues(failures)
        assert len(issues) == 2
        assert issues[0].severity == Severity.CRITICAL
        assert issues[0].code == "PII_LEAK"
        assert issues[1].severity == Severity.LOW

    def test_severity_gate_integration(self, parser):
        failures = [
            FailureRecord(FailureType.PII_LEAK, Severity.CRITICAL, "Critical leak"),
        ]
        issues = parser.to_guardrail_issues(failures)
        gate = SeverityGate()
        for issue in issues:
            gate.add(issue)
        assert gate.has_blocking is True
        with pytest.raises(SeverityBlock):
            gate.enforce()

    def test_batch_parse_multiple_outputs(self, parser):
        items = [
            {"raw_output": "Normal output 1."},
            {"raw_output": "<think>Thinking</think>Normal output 2."},
            {"raw_output": ""},
        ]
        results = parser.batch_parse(items)
        assert len(results) == 3
        assert results[0].has_failures is False
        assert any(f.failure_type == FailureType.THINK_TAG_LEAK for f in results[1].failures)
        assert any(f.failure_type == FailureType.EMPTY_OUTPUT for f in results[2].failures)

    def test_summarize_failures_statistics(self, parser):
        items = [
            {"raw_output": "Valid clean output."},
            {"raw_output": "Contains SSN 123-45-6789."},
            {"raw_output": ""},
        ]
        parsed_batch = parser.batch_parse(items)
        summary = parser.summarize_failures(parsed_batch)
        assert summary["total_outputs"] == 3
        assert summary["outputs_with_failures"] == 2
        assert summary["total_failures"] >= 2
        assert summary["severity_breakdown"]["CRITICAL"] >= 1

    def test_highest_severity_property(self, parser):
        output = "<think>Thoughts</think>Contains SSN 123-45-6789."
        parsed = parser.parse(output)
        # Should have THINK_TAG_LEAK (LOW) and PII_LEAK (HIGH/CRITICAL)
        assert parsed.highest_severity in (Severity.HIGH, Severity.CRITICAL)


# ===========================================================================
# Class 8: TestRunnerAndScriptFailures (6 tests)
# ===========================================================================
class TestRunnerAndScriptFailures:
    def test_evaluate_probe_with_pii_failure_mode(self):
        probe = {
            "id": "test_pii",
            "category": "pii_compliance",
            "input": "User SSN 123-45-6789 dispute.",
            "min_local_score": 0.50,
            "severity": "CRITICAL",
        }
        res = evaluate_probe(probe, lambda inp: "Disclosing SSN 123-45-6789 here.", judge=None)
        assert res.pii_detected is True
        assert "PII_LEAK" in res.flags

    def test_evaluate_probe_with_think_tags(self):
        probe = {
            "id": "test_think",
            "category": "summarization",
            "input": "Summarize rate hike.",
            "min_local_score": 0.50,
        }
        res = evaluate_probe(probe, lambda inp: "<think>Step 1</think>Rate hike completed.", judge=None)
        assert "THINK_TAG_LEAK" in res.flags
        assert "<think>" not in res.output_text

    def test_pre_commit_gate_passes_mock(self):
        code = run_pre_commit_gate("probes/probe_bank.yaml", min_pass_rate=0.85, max_regressed=10)
        assert code == 0

    def test_pre_commit_gate_missing_bank(self):
        code = run_pre_commit_gate("non_existent_path.yaml")
        assert code == 1

    def test_ci_gate_blocking_pii_leak(self):
        leaked_result = ProbeResult(
            probe_id="pii_leak_probe",
            category="pii_compliance",
            local_score=0.40,
            judge_score=None,
            composite=0.40,
            passed=False,
            pii_detected=True,
            severity="CRITICAL",
        )
        report = EvalRunReport.build([leaked_result])
        gate_res = evaluate_gate(report, policy=GatePolicy(block_on_pii_leak=True))
        assert gate_res.passed is False
        assert any("PII exposure" in r for r in gate_res.blocking_reasons)

    def test_cost_tracker_all_fast_passed(self):
        results = [
            ProbeResult(
                probe_id="probe_1",
                category="general",
                local_score=0.95,
                judge_score=None,
                composite=0.95,
                passed=True,
                input_text="Short input",
                output_text="Short output",
            )
        ]
        tracker = CostTracker()
        summary = tracker.calculate_savings(results)
        assert summary.probes_fast_passed_locally == 1
        assert summary.probes_escalated_to_cloud == 0
        assert summary.actual_cloud_cost_usd == 0.0
        assert summary.cost_savings_percent == 100.0
