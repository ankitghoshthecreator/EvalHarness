"""
ensemble_judge.py — Multi-Judge Ensemble Architecture with Consensus Voting & Dissent Detection.

Coordinates multiple independent LLM judges (e.g. 120B vs 20B parameter models)
to produce robust, audit-grade evaluations for financial applications.

Consensus Strategies:
  - AVERAGE: Weighted or unweighted mean composite score
  - STRICT_MINIMUM: Conservative banking policy taking worst-case score
  - MAJORITY_VOTE: Democratic pass/fail decision (>50% agreement)
  - WEIGHTED: Proportional confidence weighting across model tiers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from scorer.judge_agent import JudgeAgent, JudgeReport

logger = logging.getLogger(__name__)


class ConsensusStrategy(str, Enum):
    AVERAGE = "AVERAGE"
    STRICT_MINIMUM = "STRICT_MINIMUM"
    MAJORITY_VOTE = "MAJORITY_VOTE"
    WEIGHTED = "WEIGHTED"


@dataclass
class EnsembleDecision:
    composite: float
    passed: bool
    strategy: ConsensusStrategy
    judge_reports: dict[str, JudgeReport]
    divergence: float                        # max score - min score
    dissent_detected: bool                   # True if judges disagree significantly (>0.25)
    hallucination_conflict: bool             # True if one judge sees HIGH risk but another sees NONE
    summary_rationale: str
    pass_threshold: float = 0.70


class EnsembleJudge:
    """
    Coordinates an ensemble of multiple LLM judges.

    Usage:
        ensemble = EnsembleJudge.build_default()
        decision = ensemble.evaluate(input_text, output_text, strategy=ConsensusStrategy.AVERAGE)
    """

    def __init__(
        self,
        judges: Sequence[tuple[str, JudgeAgent, float]],
        pass_threshold: float = 0.70,
        divergence_threshold: float = 0.25,
    ):
        """
        Args:
            judges: List of (judge_name, JudgeAgent_instance, weight)
            pass_threshold: Score needed to pass
            divergence_threshold: Delta between judges that triggers dissent flag
        """
        if not judges:
            raise ValueError("Ensemble must contain at least one judge")
        self.judges = list(judges)
        self.pass_threshold = pass_threshold
        self.divergence_threshold = divergence_threshold

    @classmethod
    def build_default(cls, pass_threshold: float = 0.70) -> "EnsembleJudge":
        """Instantiate primary 120B model and secondary 20B fast model."""
        import os
        primary_model = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")
        secondary_model = os.environ.get("GROQ_SECONDARY_JUDGE", "openai/gpt-oss-20b")

        j1 = JudgeAgent(model=primary_model)
        j2 = JudgeAgent(model=secondary_model)
        return cls(
            judges=[
                ("primary_120b", j1, 0.65),
                ("secondary_20b", j2, 0.35),
            ],
            pass_threshold=pass_threshold,
        )

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        strategy: ConsensusStrategy = ConsensusStrategy.AVERAGE,
        local_fallback_score: float | None = None,
    ) -> EnsembleDecision:
        """
        Evaluate (input, output) across all judges and aggregate via consensus strategy.
        """
        reports: dict[str, JudgeReport] = {}
        scores: list[float] = []
        weights: list[float] = []

        for name, judge, weight in self.judges:
            try:
                rep = judge.evaluate(
                    input_text=input_text,
                    output_text=output_text,
                    local_fallback_score=local_fallback_score,
                )
                reports[name] = rep
                scores.append(rep.composite)
                weights.append(weight)
            except Exception as exc:
                logger.warning("Judge '%s' crashed in ensemble: %s — using fallback", name, exc)
                fb = JudgeAgent._fallback(local_fallback_score)
                reports[name] = fb
                scores.append(fb.composite)
                weights.append(weight)

        divergence = round(max(scores) - min(scores), 4) if scores else 0.0
        dissent_detected = divergence >= self.divergence_threshold

        # Detect hallucination risk conflict
        risks = {rep.hallucination_risk for rep in reports.values()}
        hallucination_conflict = ("HIGH" in risks and "NONE" in risks)

        # Apply consensus aggregation
        if strategy == ConsensusStrategy.STRICT_MINIMUM:
            composite = min(scores)
        elif strategy == ConsensusStrategy.WEIGHTED:
            total_weight = sum(weights) if sum(weights) > 0 else 1.0
            composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
        elif strategy == ConsensusStrategy.MAJORITY_VOTE:
            passes = sum(1 for s in scores if s >= self.pass_threshold)
            composite = sum(scores) / len(scores)
            passed_by_vote = passes > (len(scores) / 2.0)
            return EnsembleDecision(
                composite=round(composite, 4),
                passed=passed_by_vote,
                strategy=strategy,
                judge_reports=reports,
                divergence=divergence,
                dissent_detected=dissent_detected,
                hallucination_conflict=hallucination_conflict,
                summary_rationale=self._format_rationale(reports, strategy, dissent_detected),
                pass_threshold=self.pass_threshold,
            )
        else:  # AVERAGE default
            composite = sum(scores) / len(scores)

        composite = round(min(max(composite, 0.0), 1.0), 4)
        passed = composite >= self.pass_threshold

        return EnsembleDecision(
            composite=composite,
            passed=passed,
            strategy=strategy,
            judge_reports=reports,
            divergence=divergence,
            dissent_detected=dissent_detected,
            hallucination_conflict=hallucination_conflict,
            summary_rationale=self._format_rationale(reports, strategy, dissent_detected),
            pass_threshold=self.pass_threshold,
        )

    def _format_rationale(
        self,
        reports: dict[str, JudgeReport],
        strategy: ConsensusStrategy,
        dissent: bool,
    ) -> str:
        parts = [f"Consensus [{strategy.value}]:"]
        for name, rep in reports.items():
            parts.append(f"{name}=({rep.composite:.2f}, risk={rep.hallucination_risk})")
        if dissent:
            parts.append("[FLAG: Significant judge divergence detected]")
        return " | ".join(parts)
