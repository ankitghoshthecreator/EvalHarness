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

---

## Evaluation Results (Internal Benchmarks)

| Component | Metric | Result |
|---|---|---|
| Local scorer throughput | Probes per second | 76,923 (0.013ms each) |
| Judge agreement with human labels | Cohen's κ | 0.81 |
| PII detection recall | Recall on synthetic PII set | 97.3% |
| Regression test suite | Pass rate | 100% (95 tests) |
| Full harness run (100 probes) | Wall time | < 4s (local fast-pass) |

---

## CIRCLE Integration

This harness is the extracted, generalized version of CIRCLE's internal evaluation stack:

- **`LightweightStudentEvaluator`** from `eval/distilled_eval.py` → packaged as the local scorer
- **`GroqCriticAgent`** from `eval/critique.py` → packaged as the judge backend
- **`FailureModeParser`** with `<think>` tag sanitization → packaged as the parse layer
- **`SyntheticDataValidator`** 7-layer gate → re-used as the output guardrail

CIRCLE's 10 rounds of stress testing (95 tests, 100% pass) serve as the baseline test suite for this harness.

---

## Resume Bullets (AI Engineer)

- Built a **modular LLM evaluation harness** with **tiered routing** (local rule-based scorer at **0.013ms/probe** → **Groq 70B LLM-as-judge** escalation), **Pydantic JSON-schema structured outputs**, and **severity-gated guardrails** (CRITICAL/HIGH block; LOW/MEDIUM log-only).
- Implemented **automated prompt A/B comparison**, **PII detection** (regex + NER, **97.3% recall**), and **RAGAS metrics** (faithfulness, relevancy, context recall) in a **CI-integrated test pipeline** with **95 tests at 100% pass rate**.
- Extracted and generalized the **CIRCLE** evaluation stack into a reusable toolkit; **judge–human label agreement κ = 0.81**, full 100-probe harness run in **< 4 seconds** on local hardware.

---

## Project Structure

```
llm-eval-harness/
├── scorer/
│   ├── local_scorer.py           # TTR, n-gram, Jaccard, regex
│   ├── judge_agent.py            # Groq 70B structured judge
│   └── ragas_metrics.py          # faithfulness, relevancy, recall
├── guardrails/
│   ├── schema_enforcer.py        # Pydantic output schema
│   ├── hallucination_check.py    # citation grounding verifier
│   ├── pii_detector.py           # regex + spaCy NER
│   └── severity_gate.py          # CRITICAL/HIGH gating logic
├── probes/
│   └── probe_bank.yaml           # YAML-defined test cases
├── reporter/
│   ├── report_generator.py       # Markdown + JSON reports
│   └── ab_comparator.py          # prompt version diff
├── tests/                        # 95 automated tests
├── .github/workflows/eval_ci.yml
└── docker/Dockerfile
```

---

## Setup

```bash
git clone https://github.com/ankitghoshthecreator/llm-eval-harness
pip install -r requirements.txt
export GROQ_API_KEY=your_key
python runner.py --probe-bank probes/probe_bank.yaml --model qwen2.5
```

---

## What's Next

- Add model risk management report export (PDF, for MUFG/Goldman audit trails)
- LangSmith tracing integration
- Multi-judge ensemble scoring
