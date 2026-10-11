"""
test_cost_tracker.py — Unit tests for token economics and tiered routing savings calculations.
"""

import pytest
from reporter.cost_tracker import CostTracker, TokenCostSummary, estimate_tokens
from reporter.report_generator import ProbeResult


def _make_probe_result(pid: str, escalated: bool, input_len: int = 100, output_len: int = 100) -> ProbeResult:
    return ProbeResult(
        probe_id=pid,
        category="summarization",
        local_score=0.90 if not escalated else 0.40,
        judge_score=0.85 if escalated else None,
        composite=0.85 if escalated else 0.90,
        passed=True,
        input_text="A" * input_len,
        output_text="B" * output_len,
        latency_ms=1.5 if not escalated else 320.0,
    )


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        assert estimate_tokens("hello") >= 1

    def test_proportional(self):
        text = "word " * 100  # 500 chars -> ~125 tokens
        tokens = estimate_tokens(text)
        assert 100 <= tokens <= 150


class TestCostTracker:
    def test_empty_results(self):
        tracker = CostTracker()
        summary = tracker.calculate_savings([])
        assert summary.total_probes == 0
        assert summary.actual_cloud_cost_usd == 0.0
        assert summary.cost_savings_percent == 0.0

    def test_tiered_routing_savings(self):
        # 10 probes: 9 fast-passed locally, 1 escalated to cloud judge (90% fast pass)
        results = [_make_probe_result(f"p_{i}", escalated=False) for i in range(9)]
        results.append(_make_probe_result("p_9", escalated=True))

        tracker = CostTracker(judge_model="openai/gpt-oss-120b")
        summary = tracker.calculate_savings(results)

        assert isinstance(summary, TokenCostSummary)
        assert summary.total_probes == 10
        assert summary.probes_fast_passed_locally == 9
        assert summary.probes_escalated_to_cloud == 1
        assert summary.local_fast_pass_ratio == 0.90
        assert summary.actual_cloud_cost_usd > 0.0
        assert summary.hypothetical_all_cloud_cost_usd > summary.actual_cloud_cost_usd
        assert summary.cost_savings_usd > 0.0
        assert summary.cost_savings_percent == pytest.approx(90.0, abs=1.0)

    def test_summary_text_formatting(self):
        results = [_make_probe_result("p1", escalated=False), _make_probe_result("p2", escalated=True)]
        tracker = CostTracker()
        summary = tracker.calculate_savings(results)
        text = summary.summary()

        assert "# Token Economics & Tiered Routing Savings Report" in text
        assert "Actual Cloud Spend" in text
        assert "Net Cloud Savings" in text
