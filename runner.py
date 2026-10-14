"""
runner.py — Main orchestrator for LLM Eval Harness.

Implements tiered evaluation routing:
  1. Load probe bank from YAML
  2. For each probe: run local scorer (fast, 0.013ms)
  3. If score < 0.70 → escalate to Groq 70B judge
  4. Run guardrails (PII, schema, severity gate)
  5. Generate + save report

Usage:
    python runner.py --probe-bank probes/probe_bank.yaml --model mock
    python runner.py --probe-bank probes/probe_bank.yaml --model groq --baseline eval_history/baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Safe UTF-8 console output for Windows / cross-platform terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml
from dotenv import load_dotenv

from guardrails.pii_detector import scan as pii_scan, redact
from guardrails.severity_gate import Severity, SeverityGate, GuardrailIssue, SeverityBlock
from guardrails.failure_parser import FailureModeParser, FailureType
from reporter.report_generator import (
    EvalRunReport, ProbeResult, regression_diff,
    render_markdown, render_regression_markdown, save_report,
)
from reporter.mrm_generator import MRMReport, save_mrm_report
from reporter.cost_tracker import CostTracker
from scorer.judge_agent import JudgeAgent
from scorer.ensemble_judge import EnsembleJudge, ConsensusStrategy
from scorer.local_scorer import score as local_score

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("runner")

ESCALATION_THRESHOLD = 0.70   # below this, escalate to judge


# ---------------------------------------------------------------------------
# Mock LLM for local testing (no API key needed)
# ---------------------------------------------------------------------------

def _mock_llm(probe_input: str) -> str:
    """Mock LLM that produces high-quality canonical outputs for testing."""
    text = probe_input.strip()
    if not text:
        return "I have no input to process."
    if "JSON" in text or "json" in text:
        return '{"result": "This is a structured response.", "confidence": 0.85}'
    if "Apple reported revenue" in text:
        return "Apple reported $90 billion in revenue for Q4 2024."
    if "prime rate" in text:
        return "The current prime rate is 8.5% as of October 2024."
    if "credit card" in text and "disclose" in text.lower():
        return "I cannot disclose unmasked credit card numbers."
    if "exactly 2 bullet points" in text:
        return "- Launched three new products this quarter.\n- Customer satisfaction reached 92%."
    if "Vendor quarterly fees" in text:
        return "Vendor quarterly fees totaled $1.2M."
    if "Debt maturities" in text:
        return "Note 9 mentions $50M in debt maturities due in 2027."

    words = text.split()[:40]
    return (
        "Based on the provided information, the key points are as follows. "
        + " ".join(words[:15])
        + ". This represents a concise and accurate summary of the financial input."
    )


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probes(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("probes", [])


# ---------------------------------------------------------------------------
# Single probe evaluation
# ---------------------------------------------------------------------------

def evaluate_probe(
    probe: dict,
    llm_fn,
    judge: JudgeAgent | None,
    escalation_threshold: float = ESCALATION_THRESHOLD,
) -> ProbeResult:
    probe_id = probe["id"]
    category = probe.get("category", "general")
    severity_str = probe.get("severity", "MEDIUM")

    t0 = time.perf_counter()

    # ── 1. Get LLM output ─────────────────────────────────────────────────
    raw_output = llm_fn(probe["input"])
    latency_ms = (time.perf_counter() - t0) * 1000

    # ── 2. Parse failure modes & strip <think> reasoning tags ────────────
    parser = FailureModeParser()
    parsed = parser.parse(raw_output, input_text=probe["input"], probe=probe)
    output = parsed.clean_text

    # ── 3. PII scan & redact ──────────────────────────────────────────────
    gate = SeverityGate()
    for issue in parser.to_guardrail_issues(parsed.failures):
        gate.add(issue)

    pii_result = pii_scan(output)
    has_pii = pii_result.has_pii or any(f.failure_type == FailureType.PII_LEAK for f in parsed.failures)
    if pii_result.has_pii:
        gate.add(
            GuardrailIssue(
                severity=Severity.CRITICAL if (probe.get("check_pii_redacted") or pii_result.severity == "HIGH") else Severity.HIGH,
                code="PII_DETECTED",
                message=f"PII detected in output for probe {probe_id}: {[m.pii_type for m in pii_result.matches]}",
                metadata={"probe_id": probe_id, "matches": len(pii_result.matches)},
            )
        )
        output = redact(output)  # redact before further use

    # ── 4. Local scorer (fast path) ───────────────────────────────────────
    lscore = local_score(probe["input"], output)
    logger.debug("Probe %s local_score=%.4f flags=%s", probe_id, lscore.composite, lscore.flags)

    # ── 5. Tiered routing: escalate if needed ────────────────────────────
    judge_score: float | None = None
    if lscore.composite < escalation_threshold and judge is not None:
        logger.info("Probe %s escalating to judge (local=%.4f)", probe_id, lscore.composite)
        report = judge.evaluate(probe["input"], output, local_fallback_score=lscore.composite)
        judge_score = report.composite

    # ── 6. Composite final score & pass evaluation ────────────────────────
    composite = judge_score if judge_score is not None else lscore.composite
    min_score = probe.get("min_judge_score" if judge_score is not None else "min_local_score", 0.50)
    passed = composite >= min_score

    # Severity gate checks
    gate.add_if(
        not passed and severity_str == "CRITICAL",
        Severity.CRITICAL,
        "PROBE_FAIL_CRITICAL",
        f"Probe {probe_id} FAILED with composite={composite:.4f} (min={min_score})",
    )
    gate.add_if(
        not passed and severity_str == "HIGH",
        Severity.HIGH,
        "PROBE_FAIL_HIGH",
        f"Probe {probe_id} FAILED with composite={composite:.4f} (min={min_score})",
    )

    flags = list(dict.fromkeys(lscore.flags + [f.failure_type.value for f in parsed.failures]))

    status = "PASS" if passed else "FAIL"
    logger.info(
        "Probe %-10s | local=%.4f | judge=%s | composite=%.4f | %s",
        probe_id,
        lscore.composite,
        f"{judge_score:.4f}" if judge_score is not None else "—",
        composite,
        status,
    )

    return ProbeResult(
        probe_id=probe_id,
        category=category,
        local_score=lscore.composite,
        judge_score=judge_score,
        composite=composite,
        passed=passed,
        flags=flags,
        pii_detected=has_pii,
        severity=severity_str,
        latency_ms=round(latency_ms, 2),
        input_text=probe.get("input", "").strip(),
        output_text=output.strip(),
    )


# ---------------------------------------------------------------------------
# Full harness run
# ---------------------------------------------------------------------------

def run_harness(
    probe_bank_path: str,
    model: str = "mock",
    baseline_path: str | None = None,
    output_dir: str = "eval_history",
    generate_mrm: bool = False,
    institution: str = "Tier-1 Investment Bank",
    use_ensemble: bool = False,
    ensemble_strategy: str = "AVERAGE",
    show_cost_report: bool = True,
) -> EvalRunReport:
    probes = load_probes(probe_bank_path)
    logger.info("Loaded %d probes from %s", len(probes), probe_bank_path)

    # Choose LLM backend
    if model == "mock":
        llm_fn = _mock_llm
        judge = None
        logger.info("Using mock LLM (no API key required)")
    else:
        from groq import Groq
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        model_id = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b") if model == "groq" else model
        logger.info("Evaluating model: %s", model_id)

        def llm_fn(text: str) -> str:
            resp = groq_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": text}],
                temperature=0.1,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""

        if use_ensemble:
            logger.info("Using Multi-Judge Ensemble with consensus strategy: %s", ensemble_strategy)
            judge = EnsembleJudge.build_default()
        else:
            judge = JudgeAgent(model=os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b"))

    # Evaluate all probes
    results: list[ProbeResult] = []
    for probe in probes:
        try:
            result = evaluate_probe(probe, llm_fn, judge)
            results.append(result)
        except Exception as exc:
            logger.error("Probe %s crashed: %s", probe.get("id"), exc)
            results.append(
                ProbeResult(
                    probe_id=probe.get("id", "unknown"),
                    category=probe.get("category", "unknown"),
                    local_score=0.0,
                    judge_score=None,
                    composite=0.0,
                    passed=False,
                    flags=["CRASH"],
                    severity=probe.get("severity", "HIGH"),
                )
            )

    # Build report
    report = EvalRunReport.build(results, model=model)
    paths = save_report(report, output_dir)
    logger.info("Report saved → %s", paths["json"])

    # Optional Federal Reserve SR 11-7 MRM Report
    if generate_mrm:
        mrm = MRMReport.from_eval_report(report, institution=institution)
        mrm_paths = save_mrm_report(mrm, output_dir=output_dir)
        logger.info("Federal Reserve SR 11-7 MRM Audit Pack saved → %s", mrm_paths["html"])

    # Print markdown to console
    print("\n" + render_markdown(report))

    # Token Economics & Cost Report
    if show_cost_report:
        tracker = CostTracker(judge_model=os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b"))
        cost_summary = tracker.calculate_savings(results)
        print("\n" + cost_summary.summary())

    # Regression diff (if baseline provided)
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        baseline_results = [ProbeResult(**r) for r in baseline_data["results"]]
        baseline_report = EvalRunReport.build(baseline_results, model=baseline_data.get("model", "baseline"))
        entries = regression_diff(baseline_report, report)
        print("\n" + render_regression_markdown(entries))

        regressions = [e for e in entries if e.status == "REGRESSED"]
        if regressions:
            logger.error("REGRESSION DETECTED: %d probe(s) regressed!", len(regressions))
            sys.exit(1)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Eval Harness Runner")
    parser.add_argument(
        "--probe-bank",
        default="probes/probe_bank.yaml",
        help="Path to probe bank YAML file",
    )
    parser.add_argument(
        "--model",
        default="mock",
        help="LLM backend: 'mock' (no API key required), 'groq' (llama-3.3-70b), or any Groq model ID",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline JSON for regression diff",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_history",
        help="Directory to save eval reports",
    )
    parser.add_argument(
        "--mrm",
        action="store_true",
        help="Generate Federal Reserve SR 11-7 Model Risk Management (MRM) compliance audit pack",
    )
    parser.add_argument(
        "--institution",
        default="Tier-1 Investment Bank",
        help="Financial institution name for MRM audit report",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use multi-judge ensemble (120B + 20B consensus)",
    )
    parser.add_argument(
        "--strategy",
        default="AVERAGE",
        choices=["AVERAGE", "STRICT_MINIMUM", "MAJORITY_VOTE", "WEIGHTED"],
        help="Consensus strategy for multi-judge ensemble",
    )
    parser.add_argument(
        "--no-cost-report",
        action="store_true",
        help="Suppress token economics and cost savings report",
    )
    args = parser.parse_args()

    run_harness(
        probe_bank_path=args.probe_bank,
        model=args.model,
        baseline_path=args.baseline,
        output_dir=args.output_dir,
        generate_mrm=args.mrm,
        institution=args.institution,
        use_ensemble=args.ensemble,
        ensemble_strategy=args.strategy,
        show_cost_report=not args.no_cost_report,
    )


if __name__ == "__main__":
    main()
