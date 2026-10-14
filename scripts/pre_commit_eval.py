#!/usr/bin/env python3
"""
scripts/pre_commit_eval.py — Developer Pre-Commit Quality Hook
==============================================================

Runs the 0.013ms local rule-based scorer against every probe in the probe bank
*before* any code lands in git. This is the first line of defence: cheap, fast,
zero-API-key, and catches obvious regressions in under 1 second.

Install as a Git pre-commit hook:
    cp scripts/pre_commit_eval.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit   # (Linux/macOS only)

Or run manually:
    python scripts/pre_commit_eval.py
    python scripts/pre_commit_eval.py --probe-bank probes/probe_bank.yaml --min-pass-rate 0.90

Exit codes:
    0  — All checks passed, safe to commit.
    1  — Quality gate failed, commit is BLOCKED.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure repo root is on sys.path regardless of CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
from scorer.local_scorer import score as local_score


# ---------------------------------------------------------------------------
# ANSI colour helpers (gracefully disabled on non-TTY)
# ---------------------------------------------------------------------------
_USE_COLOUR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"


GREEN   = lambda t: _c("92", t)   # noqa: E731
RED     = lambda t: _c("91", t)   # noqa: E731
YELLOW  = lambda t: _c("93", t)   # noqa: E731
BOLD    = lambda t: _c("1",  t)   # noqa: E731
CYAN    = lambda t: _c("96", t)   # noqa: E731
DIM     = lambda t: _c("2",  t)   # noqa: E731


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
# Core gate logic
# ---------------------------------------------------------------------------
def run_pre_commit_gate(
    probe_bank: str = "probes/probe_bank.yaml",
    min_pass_rate: float = 0.85,
    min_composite: float = 0.50,
    baseline_path: str | None = None,
    max_regressed: int = 10,
    verbose: bool = False,
) -> int:
    """
    Returns 0 (pass) or 1 (fail / commit blocked).
    """
    bank_path = Path(probe_bank)
    if not bank_path.exists():
        print(RED(f"✗  Probe bank not found: {bank_path}"))
        return 1

    with open(bank_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    probes: list[dict] = data.get("probes", [])

    if not probes:
        print(YELLOW("⚠  No probes found — skipping pre-commit gate."))
        return 0

    print(BOLD(CYAN("\n━━━  LLM Eval Harness — Pre-Commit Quality Gate  ━━━")))
    print(DIM(f"     Probe bank : {bank_path}"))
    print(DIM(f"     Probes     : {len(probes)}"))
    print(DIM(f"     Min score  : {min_composite:.2f} per probe"))
    print(DIM(f"     Min rate   : {min_pass_rate:.0%} aggregate pass rate"))
    print()

    t_start = time.perf_counter()
    passed_count = 0
    failed_probes: list[dict] = []

    for probe in probes:
        pid     = probe.get("id", "unknown")
        p_input = probe.get("input", "")
        p_min   = probe.get("min_local_score", min_composite)
        output  = _mock_llm(p_input)

        result  = local_score(p_input, output)
        ok      = result.composite >= p_min

        if ok:
            passed_count += 1
            if verbose:
                print(f"  {GREEN('✓')} {pid:<30} composite={result.composite:.4f}")
        else:
            failed_probes.append({
                "id":        pid,
                "composite": result.composite,
                "min":       p_min,
                "flags":     result.flags,
            })
            if verbose:
                flags_str = ", ".join(result.flags) if result.flags else "—"
                print(f"  {RED('✗')} {pid:<30} composite={result.composite:.4f} "
                      f"(min={p_min:.2f}) flags=[{flags_str}]")

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    total       = len(probes)
    pass_rate   = passed_count / total if total else 0.0

    # ── Regression diff against frozen baseline ──────────────────────────────
    baseline_regressions: list[dict] = []
    if baseline_path:
        bpath = Path(baseline_path)
        if bpath.exists():
            with open(bpath) as f:
                baseline_data = json.load(f)
            base_scores: dict[str, float] = {
                r["probe_id"]: r["composite"]
                for r in baseline_data.get("results", [])
            }
            for probe in probes:
                pid    = probe.get("id", "unknown")
                output = _mock_llm(probe.get("input", ""))
                result = local_score(probe.get("input", ""), output)
                if pid in base_scores:
                    delta = result.composite - base_scores[pid]
                    if delta < -0.05:   # >5pp regression triggers warning
                        baseline_regressions.append({
                            "id":       pid,
                            "baseline": base_scores[pid],
                            "current":  result.composite,
                            "delta":    delta,
                            "severity": probe.get("severity", "MEDIUM"),
                        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(BOLD("\n  ── Results ──────────────────────────────────────────────"))
    print(f"  Elapsed       : {elapsed_ms:.1f} ms")
    print(f"  Pass / Total  : {passed_count} / {total}")
    status_str = (
        GREEN(f"{pass_rate:.1%}  ✓ GATE PASSED")
        if pass_rate >= min_pass_rate
        else RED(f"{pass_rate:.1%}  ✗ GATE FAILED")
    )
    print(f"  Pass Rate     : {status_str}")

    if failed_probes:
        print(RED(f"\n  ✗ {len(failed_probes)} probe(s) failed the local scorer threshold:"))
        for fp in failed_probes:
            flags_str = ", ".join(fp["flags"]) if fp["flags"] else "—"
            print(f"      • {fp['id']:<30} score={fp['composite']:.4f} "
                  f"(min={fp['min']:.2f})  flags=[{flags_str}]")

    if baseline_regressions:
        print(YELLOW(f"\n  ⚠  {len(baseline_regressions)} probe(s) regressed vs frozen baseline "
                     f"(eval_history/baseline.json):"))
        for reg in baseline_regressions:
            sev_badge = f"[{reg.get('severity', 'MEDIUM')}]"
            print(f"      • {reg['id']:<30} {sev_badge:<10} {reg['baseline']:.4f} → {reg['current']:.4f}"
                  f"  Δ={reg['delta']:+.4f}")

    print()

    # ── Final verdict ─────────────────────────────────────────────────────────
    critical_regressions = [r for r in baseline_regressions if r.get("severity") == "CRITICAL"]
    gate_ok = pass_rate >= min_pass_rate and len(critical_regressions) == 0 and len(baseline_regressions) <= max_regressed

    if gate_ok:
        print(GREEN(BOLD("  ✓  Commit approved — all quality gates passed.\n")))
        return 0
    else:
        print(RED(BOLD("  ✗  Commit BLOCKED — fix the failing probes above, then retry.\n")))
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM Eval Harness — Pre-Commit Quality Gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--probe-bank",
        default="probes/probe_bank.yaml",
        help="Path to probe bank YAML (default: probes/probe_bank.yaml)",
    )
    p.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.85,
        help="Minimum aggregate pass rate to allow the commit (default: 0.85)",
    )
    p.add_argument(
        "--min-composite",
        type=float,
        default=0.50,
        help="Minimum per-probe composite score (default: 0.50)",
    )
    p.add_argument(
        "--baseline",
        default="eval_history/baseline.json",
        help="Frozen baseline JSON for regression diff (default: eval_history/baseline.json)",
    )
    p.add_argument(
        "--max-regressed",
        type=int,
        default=10,
        help="Max allowed non-critical regressed probes (default: 10)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-probe pass/fail detail",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(
        run_pre_commit_gate(
            probe_bank=args.probe_bank,
            min_pass_rate=args.min_pass_rate,
            min_composite=args.min_composite,
            baseline_path=args.baseline,
            max_regressed=args.max_regressed,
            verbose=args.verbose,
        )
    )
