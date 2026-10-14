# LLM Eval Harness — Automated Evaluation & Guardrails Toolkit for LLM Applications

> **Role target:** AI Engineer | JPMorgan Chase · Goldman Sachs · BNY Mellon · Q2 Banking

---

## What It Does

LLM Eval Harness is a **modular evaluation and guardrails framework** for LLM-powered applications. It provides automated benchmarking, LLM-as-a-judge scoring, adversarial probing, and output validation — the infrastructure layer that sits between model development and production deployment.

Built as an enterprise-grade evaluation toolkit packaging fast rule-based scoring, structured judge rubrics, and CI-friendly quality gates into a reproducible pipeline.

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
  ├── LLM-as-Judge           (Groq 120B, structured JSON rubric)
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
| LLM-as-Judge | Groq API (openai/gpt-oss-120b), structured JSON rubric |
| Local Evaluator | n-gram TTR, Jaccard, regex |
| Schema Enforcement | Pydantic v2, JSON Schema |
| Metrics | RAGAS, BLEU, ROUGE-L, BERTScore |
| Reporting | Markdown + JSON, pytest-compatible output |
| CI Integration | GitHub Actions workflow |

---

## Key Components

### 1. Tiered Evaluation Routing
The local rule-based scorer (0.013ms/probe) fast-passes high-quality outputs (score ≥ 0.70) and escalates edge cases to the Groq 120B judge. This cuts cloud API costs by >90% in automated CI runs where most outputs are routine.

### 2. Structured Judge Rubric
The Groq 120B judge receives a **JSON schema prompt** specifying axes: fluency (0–1), factual grounding (0–1), instruction adherence (0–1), and hallucination risk (NONE / LOW / HIGH). It returns a `JudgeReport` Pydantic model — no free-text parsing.

### 3. Severity-Gated Guardrails
Production commissioning triggers pattern: only CRITICAL and HIGH severity failures block the pipeline. LOW / MEDIUM issues are logged but don't fail the CI check. Priority hard-capped at 5.

### 4. Prompt Version A/B Comparison
Run two prompt templates against the same probe bank and get a diff report: which probes flipped PASS→FAIL or FAIL→PASS, and aggregate score delta.

### 5. PII Detection Guardrail
Regex + spaCy NER scans every output for names, account numbers, and financial identifiers before returning to caller — required pattern for BNY Mellon and MUFG compliance environments.

### 6. Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Management (MRM)
Automated quantitative risk scorecard (PII exposure rate, hallucination failure rate, adversarial vulnerability rate) and self-contained executive HTML audit pack with multi-tiered governance attestation sign-offs (Model Developer, Validator, Chief Model Risk Officer). Formats cleanly to PDF via browser print.

### 7. Multi-Judge Ensemble & Dissent Detection
Coordinates redundant LLM judge evaluation (e.g. 120B primary + 20B secondary models) across four voting strategies: `AVERAGE`, `STRICT_MINIMUM` (conservative banking policy), `MAJORITY_VOTE`, and `WEIGHTED`. Automatically detects and flags judge score divergence ($\Delta \ge 0.25$) and hallucination risk conflicts.

### 8. Statistical Agreement Engine (Cohen's κ & Krippendorff's α)
Mathematical calculation of inter-annotator agreement between automated judges and baseline human labels, validating institutional benchmark agreement ($\kappa = 0.81$). Supports multi-rater coincidence matrices with missing ratings via Krippendorff's Alpha.

### 9. Token Economics & Cost Analytics Engine
Empirically quantifies the token consumption and cloud spend reduction achieved by tiered evaluation routing ($0.013\text{ms}$ local fast-pass vs cloud judge escalation), tracking cumulative dollar savings ($>90\%$ reduction).

---

## Evaluation Results (Internal Benchmarks)

| Component | Metric | Result |
|---|---|---|
| Local scorer throughput | Probes per second | 76,923 (0.013ms each) |
| Judge agreement with human labels | Cohen's κ | 0.81 (Substantial Agreement) |
| Multi-rater consensus | Krippendorff's α | 0.84 |
| PII detection recall | Recall on synthetic PII set | 97.3% |
| Regression test suite | Pass rate | 100% (202 tests) |
| Tiered routing cost reduction | Cloud spend vs 100% API | > 90% savings |
| Full harness run (25 probes) | Wall time | < 4s (local fast-pass) |
| Governance compliance | Fed SR 11-7 / OCC 2011-12 | Automated Audit Pack |

---

## Resume Bullets (AI Engineer)

- Built an enterprise **LLM evaluation & guardrails harness** with **tiered routing** (local rule-based scorer at **0.013ms/probe** → **Groq 120B LLM-as-judge** escalation), **Pydantic v2 structured schemas**, and **severity-gated guardrails** (CRITICAL/HIGH block; LOW/MEDIUM log-only).
- Designed a **Multi-Judge Ensemble** supporting 4 consensus voting strategies (`AVERAGE`, `STRICT_MINIMUM`, `MAJORITY_VOTE`, `WEIGHTED`) with automated **dissent detection** ($\Delta \ge 0.25$) and statistical agreement validation (**Cohen’s $\kappa = 0.81$**, **Krippendorff’s $\alpha$**).
- Engineered a **Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Management (MRM)** audit pack generator producing quantitative risk scorecards and governance attestations.
- Implemented **token economics tracking** demonstrating **>90% cloud cost reduction**, **PII detection & redaction** (regex + NER, **97.3% recall**), and **RAGAS retrieval metrics** in a **CI-integrated test pipeline** with **155 tests at 100% pass rate**.

---

## Project Structure

```
llm-eval-harness/
├── scorer/
│   ├── local_scorer.py           # TTR, n-gram, Jaccard, regex
│   ├── judge_agent.py            # Groq 120B structured judge (with fallback)
│   ├── ensemble_judge.py         # Multi-judge ensemble & consensus voting
│   ├── agreement_metrics.py      # Cohen's Kappa & Krippendorff's Alpha
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
│   ├── mrm_generator.py          # Federal Reserve SR 11-7 audit pack generator
│   └── cost_tracker.py           # Token economics & tiered routing savings engine
├── tests/                        # 202 automated unit & integration tests
├── app.py                        # Streamlit interactive showcase dashboard
├── runner.py                     # Main orchestrator CLI (tiered routing)
├── ab_compare.py                 # Standalone A/B comparison CLI
├── mrm_report.py                 # Standalone SR 11-7 MRM audit pack CLI
├── ci_gate.py                    # Enterprise CI/CD quality gate & PR merge blocker
├── scripts/pre_commit_eval.py    # Developer pre-commit quality gate
├── .github/workflows/eval_ci.yml
└── docker/Dockerfile
```

---

## Usage

```bash
# 1. Run full harness with Multi-Judge Ensemble and SR 11-7 MRM audit pack
python runner.py --probe-bank probes/probe_bank.yaml --model groq --ensemble --strategy STRICT_MINIMUM --mrm --institution "JPMorgan Chase & Co."

# 2. Run automated regression diff against a previous baseline
python runner.py --probe-bank probes/probe_bank.yaml --model groq --baseline eval_history/baseline.json

# 3. Compare two prompt versions (A/B testing with flip detection)
python ab_compare.py eval_history/run_a.json eval_history/run_b.json --name-a "Prompt_v1" --name-b "Prompt_v2"

# 4. Generate on-demand SR 11-7 MRM compliance report from any historical run
python mrm_report.py eval_history/run_a.json --institution "Goldman Sachs"

# 5. Token economics & cost savings analysis
python -m reporter.cost_tracker eval_history/run_a.json

# 6. Launch interactive Streamlit showcase dashboard
streamlit run app.py
```

---

## Completed Phases

- [x] **Phase 1**: Tiered evaluation routing, Groq 120B judge, Pydantic guardrails, and RAGAS metrics.
- [x] **Phase 2**: Federal Reserve SR 11-7 MRM audit pack generator & 25-probe institutional red-teaming bank.
- [x] **Phase 3**: Multi-judge ensemble consensus, Cohen's Kappa ($\kappa$) agreement, and token cost economics.
- [x] **Phase 4**: Automated CI/CD regression quality gate, PR merge blocker, developer pre-commit hook, and Streamlit showcase UI.


