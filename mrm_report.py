"""
mrm_report.py — CLI to generate Federal Reserve SR 11-7 Model Risk Management (MRM) audit reports.

Usage:
    python mrm_report.py eval_history/20261011T184128Z.json
    python mrm_report.py eval_history/20261011T184128Z.json --institution "JPMorgan Chase & Co." --output-dir eval_history
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reporter.mrm_generator import MRMReport, save_mrm_report
from reporter.report_generator import EvalRunReport, ProbeResult


def main():
    parser = argparse.ArgumentParser(
        description="Federal Reserve SR 11-7 Model Risk Management (MRM) Audit Pack Generator"
    )
    parser.add_argument("report_json", help="Path to evaluation run JSON report")
    parser.add_argument(
        "--institution",
        default="Tier-1 Investment Bank",
        help="Target Financial Institution Name (e.g., 'Goldman Sachs', 'JPMorgan Chase')",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_history",
        help="Directory to save the generated MRM audit pack",
    )
    args = parser.parse_args()

    with open(args.report_json, encoding="utf-8") as f:
        data = json.load(f)

    results = [ProbeResult(**r) for r in data.get("results", [])]
    eval_report = EvalRunReport(
        run_id=data.get("run_id", "unknown"),
        timestamp=data.get("timestamp", ""),
        model=data.get("model", "unknown"),
        total_probes=data.get("total_probes", len(results)),
        passed=data.get("passed", sum(1 for r in results if r.passed)),
        failed=data.get("failed", sum(1 for r in results if not r.passed)),
        pass_rate=data.get("pass_rate", 0.0),
        avg_local_score=data.get("avg_local_score", 0.0),
        avg_judge_score=data.get("avg_judge_score", 0.0),
        results=results,
        metadata=data.get("metadata", {}),
    )

    mrm = MRMReport.from_eval_report(eval_report, institution=args.institution)
    paths = save_mrm_report(mrm, output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("FEDERAL RESERVE SR 11-7 MODEL RISK MANAGEMENT AUDIT PACK GENERATED")
    print("=" * 70)
    print(f"Run ID:             {mrm.run_id}")
    print(f"Model ID:           {mrm.model_id}")
    print(f"Institution:        {mrm.institution}")
    print(f"Overall Pass Rate:  {mrm.metrics.pass_rate * 100:.1f}%")
    print(f"Risk Rating:        {mrm.metrics.overall_risk_rating}")
    print(f"PII Exposure Rate:  {mrm.metrics.pii_exposure_rate * 100:.1f}%")
    print(f"Hallucination Rate: {mrm.metrics.hallucination_failure_rate * 100:.1f}%")
    print("-" * 70)
    print(f"Audit HTML:         {paths['html']}")
    print(f"Audit JSON:         {paths['json']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
