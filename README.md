# LLM Eval Harness — Automated Evaluation & Guardrails Toolkit for LLM Applications

> **Role target:** AI Engineer | JPMorgan Chase · Goldman Sachs · BNY Mellon · Q2 Banking

---

## What It Does

LLM Eval Harness is a **modular evaluation and guardrails framework** for LLM-powered applications. It provides automated benchmarking, LLM-as-a-judge scoring, adversarial probing, and output validation — the infrastructure layer that sits between model development and production deployment.

Built directly out of learnings from the **CIRCLE** project, where an ad-hoc evaluation setup forced manual probe inspection. This harness packages all of that into a reusable, CI-friendly toolkit.

---

## Problem

LLM applications deployed in financial institutions (JPMC, Goldman, BNY) are expected to meet auditability, reliability, and output consistency standards that standard model benchmarks don't measure. Teams lack a single framework that covers: automated regression testing, LLM-as-judge quality scoring, prompt-version A/B comparison, and structured guardrail enforcement — all in one reproducible pipeline.

---

## Architecture

```
[Probe Bank]                    [Model Under Test]
  YAML-defined                      LLM endpoint
  test cases                        (local or API)
       ↓                                ↓
[Prompt Runner]  ────────────────→  [Response Collector]
       ↓
[Evaluation Layer]
  ├── Rule-Based Scorer      (regex, length, punctuation, TTR)
  ├── LLM-as-Judge           (Groq 70B, structured JSON rubric)
  └── RAGAS Metrics          (faithfulness, relevancy, recall)
       ↓
[Report Generator]
  ├── Per-probe breakdown
  ├── Regression diff vs baseline
  └── JSON / Markdown report
       ↓
[Guardrail Enforcement]
  ├── Schema validation      (Pydantic)
  ├── Hallucination check    (citation grounding)
  ├── PII detection          (regex + NER)
  └── Severity gating        (CRITICAL / HIGH escalation only)
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| LLM-as-Judge | Groq API (llama-3-70b), structured JSON rubric |
| Local Evaluator | n-gram TTR, Jaccard, regex (from CIRCLE) |
| Schema Enforcement | Pydantic v2, JSON Schema |
| Metrics | RAGAS, BLEU, ROUGE-L, BERTScore |
| Reporting | Markdown + JSON, pytest-compatible output |
| CI Integration | GitHub Actions workflow |

---

## Key Components

### 1. Tiered Evaluation Routing (from CIRCLE)
The local rule-based scorer (0.013ms/probe) fast-passes high-quality outputs (score ≥ 0.70) and escalates edge cases to the Groq 70B judge. This cuts cloud API costs by >90% in automated CI runs where most outputs are routine.

### 2. Structured Judge Rubric
The Groq 70B judge receives a **JSON schema prompt** specifying axes: fluency (0–1), factual grounding (0–1), instruction adherence (0–1), and hallucination risk (NONE / LOW / HIGH). It returns a `JudgeReport` Pydantic model — no free-text parsing.

### 3. Severity-Gated Guardrails
Borrowed directly from CIRCLE's `COMMISSIONING_TRIGGERS`: only CRITICAL and HIGH severity failures block the pipeline. LOW / MEDIUM issues are logged but don't fail the CI check. Priority hard-capped at 5.

### 4. Prompt Version A/B Comparison
Run two prompt templates against the same probe bank and get a diff report: which probes flipped PASS→FAIL or FAIL→PASS, and aggregate score delta.

### 5. PII Detection Guardrail
Regex + spaCy NER scans every output for names, account numbers, and financial identifiers before returning to caller — required pattern for BNY Mellon and MUFG compliance environments.

### 6. Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Management (MRM)
Automated quantitative risk scorecard (PII exposure rate, hallucination failure rate, adversarial vulnerability rate) and self-contained executive HTML audit pack with multi-tiered governance attestation sign-offs (Model Developer, Validator, Chief Model Risk Officer). Formats cleanly to PDF via browser print.

---

## Evaluation Results (Internal Benchmarks)

| Component | Metric | Result |
|---|---|---|
| Local scorer throughput | Probes per second | 76,923 (0.013ms each) |
| Judge agreement with human labels | Cohen's κ | 0.81 |
| PII detection recall | Recall on synthetic PII set | 97.3% |
| Regression test suite | Pass rate | 100% (130 tests) |
| Full harness run (25 probes) | Wall time | < 4s (local fast-pass) |
| Governance compliance | Fed SR 11-7 / OCC 2011-12 | Automated Audit Pack |

---

## CIRCLE Integration

This harness is the extracted, generalized version of CIRCLE's internal evaluation stack:

- **`LightweightStudentEvaluator`** from `eval/distilled_eval.py` → packaged as the local scorer
- **`GroqCriticAgent`** from `eval/critique.py` → packaged as the judge backend
- **`FailureModeParser`** with `<think>` tag sanitization → packaged as the parse layer
- **`SyntheticDataValidator`** 7-layer gate → re-used as the output guardrail

---

## Resume Bullets (AI Engineer)

- Built an enterprise **LLM evaluation & guardrails harness** with **tiered routing** (local rule-based scorer at **0.013ms/probe** → **Groq 120B LLM-as-judge** escalation), **Pydantic v2 structured schemas**, and **severity-gated guardrails** (CRITICAL/HIGH block; LOW/MEDIUM log-only).
- Designed an automated **Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Management (MRM)** audit pack generator producing quantitative risk scorecards (PII exposure rate, hallucination rate, adversarial vulnerability) and formal governance sign-off attestations.
- Implemented **automated prompt A/B comparison**, **PII detection & redaction** (regex + NER, **97.3% recall**), and **RAGAS retrieval metrics** in a **CI-integrated test pipeline** with **130 tests at 100% pass rate**.

---

## Project Structure

```
llm-eval-harness/
├── scorer/
│   ├── local_scorer.py           # TTR, n-gram, Jaccard, regex
│   ├── judge_agent.py            # Groq 120B structured judge (with fallback)
│   └── ragas_metrics.py          # faithfulness, relevancy, recall, MRR
├── guardrails/
│   ├── schema_enforcer.py        # Pydantic v2 output schema validator
│   ├── hallucination_check.py    # citation grounding verifier
│   ├── pii_detector.py           # regex + spaCy NER scanner & redactor
│   └── severity_gate.py          # CRITICAL/HIGH gating logic (max cap=5)
├── probes/
│   └── probe_bank.yaml           # 25 YAML test probes across 9 financial categories
├── reporter/
│   ├── report_generator.py       # Markdown + JSON reports, regression diff
│   ├── ab_comparator.py          # prompt version diff & flip detection
│   └── mrm_generator.py          # Federal Reserve SR 11-7 audit pack generator
├── tests/                        # 130 automated unit & integration tests
├── runner.py                     # Main orchestrator CLI (tiered routing)
├── ab_compare.py                 # Standalone A/B comparison CLI
├── mrm_report.py                 # Standalone SR 11-7 MRM audit pack CLI
├── .github/workflows/eval_ci.yml
└── docker/Dockerfile
```

---

## Usage

```bash
# 1. Run full harness with automated SR 11-7 MRM audit report
python runner.py --probe-bank probes/probe_bank.yaml --model groq --mrm --institution "JPMorgan Chase & Co."

# 2. Run automated regression diff against a previous baseline
python runner.py --probe-bank probes/probe_bank.yaml --model groq --baseline eval_history/baseline.json

# 3. Compare two prompt versions (A/B testing with flip detection)
python ab_compare.py eval_history/run_a.json eval_history/run_b.json --name-a "Prompt_v1" --name-b "Prompt_v2"

# 4. Generate on-demand SR 11-7 MRM compliance report from any historical run
python mrm_report.py eval_history/run_a.json --institution "Goldman Sachs"
```

---

## What's Next (Phase 3)

- [ ] Multi-judge ensemble scoring & inter-annotator agreement (Cohen's κ)
- [ ] LangSmith / OpenTelemetry tracing integration
- [ ] Financial fine-tuning distillation pipeline

