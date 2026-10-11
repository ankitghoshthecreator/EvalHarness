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

import yaml
from dotenv import load_dotenv

from guardrails.pii_detector import scan as pii_scan, redact
from guardrails.severity_gate import Severity, SeverityGate, GuardrailIssue, SeverityBlock
from reporter.report_generator import (
    EvalRunReport, ProbeResult, regression_diff,
    render_markdown, render_regression_markdown, save_report,
)
from reporter.mrm_generator import MRMReport, save_mrm_report
from scorer.judge_agent import JudgeAgent
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
    """Simple echo-style mock that produces reasonable outputs for testing."""
    words = probe_input.strip().split()[:40]
    if "JSON" in probe_input or "json" in probe_input:
        return '{"result": "This is a structured response.", "confidence": 0.85}'
    return (
        "Based on the provided information, the key points are as follows. "
        + " ".join(words[:20])
        + ". This represents a concise and accurate summary of the input."
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
    output = llm_fn(probe["input"])
    latency_ms = (time.perf_counter() - t0) * 1000

    # ── 2. PII scan (before any storage / further processing) ─────────────
    gate = SeverityGate()
    pii_result = pii_scan(output)
    if pii_result.has_pii:
        gate.add(
            GuardrailIssue(
                severity=Severity.CRITICAL if pii_result.severity == "HIGH" else Severity.HIGH,
                code="PII_DETECTED",
                message=f"PII detected in output for probe {probe_id}: {[m.pii_type for m in pii_result.matches]}",
                metadata={"probe_id": probe_id, "matches": len(pii_result.matches)},
            )
        )
        output = redact(output)  # redact before further use

    # ── 3. Local scorer (fast path) ───────────────────────────────────────
    lscore = local_score(probe["input"], output)
    logger.debug("Probe %s local_score=%.4f flags=%s", probe_id, lscore.composite, lscore.flags)

    # ── 4. Tiered routing: escalate if needed ────────────────────────────
    judge_score: float | None = None
    if lscore.composite < escalation_threshold and judge is not None:
        logger.info("Probe %s escalating to judge (local=%.4f)", probe_id, lscore.composite)
        report = judge.evaluate(probe["input"], output, local_fallback_score=lscore.composite)
        judge_score = report.composite

    # ── 5. Composite final score ──────────────────────────────────────────
    composite = judge_score if judge_score is not None else lscore.composite
    min_score = probe.get("min_local_score", 0.50)
    passed = composite >= min_score

    # ── 6. Severity gate checks ───────────────────────────────────────────
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
        flags=lscore.flags,
        pii_detected=pii_result.has_pii,
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
    args = parser.parse_args()

    run_harness(
        probe_bank_path=args.probe_bank,
        model=args.model,
        baseline_path=args.baseline,
        output_dir=args.output_dir,
        generate_mrm=args.mrm,
        institution=args.institution,
    )


if __name__ == "__main__":
    main()
