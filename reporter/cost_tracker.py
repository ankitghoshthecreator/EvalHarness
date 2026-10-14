"""
cost_tracker.py — Token Economics & Tiered Routing Savings Engine.

Quantifies token consumption, cloud API costs, and empirical cost reduction
achieved by the tiered evaluation routing architecture (0.013ms local fast-pass
vs cloud LLM-as-a-judge escalation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from reporter.report_generator import ProbeResult


# Standard reference pricing per 1,000,000 tokens (USD)
PRICING_TABLE: dict[str, dict[str, float]] = {
    "openai/gpt-oss-120b": {"prompt": 0.59, "completion": 0.79},
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
    "openai/gpt-oss-20b": {"prompt": 0.08, "completion": 0.08},
    "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
    "local_scorer": {"prompt": 0.00, "completion": 0.00},
}


def estimate_tokens(text: str) -> int:
    """Standard ~4 characters per token heuristic for fast offline calculation."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class TokenCostSummary:
    total_probes: int
    probes_fast_passed_locally: int
    probes_escalated_to_cloud: int
    local_fast_pass_ratio: float
    total_tokens_consumed: int
    actual_cloud_cost_usd: float
    hypothetical_all_cloud_cost_usd: float
    cost_savings_usd: float
    cost_savings_percent: float
    avg_latency_ms: float

    def summary(self) -> str:
        lines = [
            "# Token Economics & Tiered Routing Savings Report",
            "",
            f"**Total Probes Evaluated:** {self.total_probes}",
            f"  - Fast-passed locally (0.013ms): {self.probes_fast_passed_locally} ({self.local_fast_pass_ratio * 100:.1f}%)",
            f"  - Escalated to Cloud Judge:     {self.probes_escalated_to_cloud} ({(1.0 - self.local_fast_pass_ratio) * 100:.1f}%)",
            "",
            "## Cost Breakdown",
            f"  - Actual Cloud Spend (Tiered):        ${self.actual_cloud_cost_usd:.6f}",
            f"  - Baseline Spend (100% Cloud Judge):  ${self.hypothetical_all_cloud_cost_usd:.6f}",
            f"  - **Net Cloud Savings:**              **${self.cost_savings_usd:.6f} ({self.cost_savings_percent:.1f}%)**",
            "",
            f"**Average Latency per Probe:** {self.avg_latency_ms:.2f} ms",
        ]
        return "\n".join(lines)


class CostTracker:
    """
    Calculates token counts, cloud expenditure, and routing savings.
    """

    def __init__(self, judge_model: str = "openai/gpt-oss-120b"):
        self.judge_model = judge_model
        pricing = PRICING_TABLE.get(judge_model, PRICING_TABLE["openai/gpt-oss-120b"])
        self.prompt_price_per_m = pricing["prompt"]
        self.completion_price_per_m = pricing["completion"]

    def calculate_savings(self, results: Sequence[ProbeResult]) -> TokenCostSummary:
        total = len(results)
        if total == 0:
            return TokenCostSummary(
                total_probes=0,
                probes_fast_passed_locally=0,
                probes_escalated_to_cloud=0,
                local_fast_pass_ratio=0.0,
                total_tokens_consumed=0,
                actual_cloud_cost_usd=0.0,
                hypothetical_all_cloud_cost_usd=0.0,
                cost_savings_usd=0.0,
                cost_savings_percent=0.0,
                avg_latency_ms=0.0,
            )

        fast_passed = sum(1 for r in results if r.judge_score is None)
        escalated = total - fast_passed

        total_tokens = 0
        actual_cost = 0.0
        baseline_cost = 0.0
        latencies = [r.latency_ms for r in results]

        for r in results:
            prompt_tok = estimate_tokens(r.input_text)
            comp_tok = estimate_tokens(r.output_text)
            probe_tok = prompt_tok + comp_tok
            total_tokens += probe_tok

            # Cost per probe on judge model
            probe_cost = (
                (prompt_tok * self.prompt_price_per_m / 1_000_000)
                + (comp_tok * self.completion_price_per_m / 1_000_000)
            )

            # Baseline assumes ALL probes hit cloud judge
            baseline_cost += probe_cost

            # Actual only incurs cost if escalated to judge
            if r.judge_score is not None:
                actual_cost += probe_cost

        savings_usd = max(0.0, baseline_cost - actual_cost)
        savings_pct = (savings_usd / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

        return TokenCostSummary(
            total_probes=total,
            probes_fast_passed_locally=fast_passed,
            probes_escalated_to_cloud=escalated,
            local_fast_pass_ratio=round(fast_passed / total, 4),
            total_tokens_consumed=total_tokens,
            actual_cloud_cost_usd=round(actual_cost, 6),
            hypothetical_all_cloud_cost_usd=round(baseline_cost, 6),
            cost_savings_usd=round(savings_usd, 6),
            cost_savings_percent=round(savings_pct, 2),
            avg_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        )


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="LLM Eval Harness — Token Economics & Cost Analytics")
    parser.add_argument("report_json", help="Path to evaluation run JSON report")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Judge model name for pricing lookup")
    args = parser.parse_args()

    with open(args.report_json, encoding="utf-8") as f:
        data = json.load(f)

    results = [ProbeResult(**r) for r in data.get("results", [])]
    tracker = CostTracker(judge_model=args.model)
    summary = tracker.calculate_savings(results)
    print("\n" + summary.summary() + "\n")


if __name__ == "__main__":
    main()
