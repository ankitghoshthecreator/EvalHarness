"""
test_ensemble_judge.py — Unit tests for multi-judge ensemble architecture and consensus voting.
"""

from unittest.mock import MagicMock
import pytest
from scorer.ensemble_judge import EnsembleJudge, EnsembleDecision, ConsensusStrategy
from scorer.judge_agent import JudgeAgent, JudgeReport


def _mock_judge(score: float, risk: str = "NONE") -> JudgeAgent:
    j = JudgeAgent(api_key="fake")
    report = JudgeReport(
        fluency=score,
        factual_grounding=score,
        instruction_adherence=score,
        hallucination_risk=risk,
    )
    j.evaluate = MagicMock(return_value=report)
    return j


class TestEnsembleJudgeStrategies:
    def test_average_strategy(self):
        j1 = _mock_judge(0.90)
        j2 = _mock_judge(0.80)
        ensemble = EnsembleJudge([("j1", j1, 0.5), ("j2", j2, 0.5)])

        decision = ensemble.evaluate("input", "output", strategy=ConsensusStrategy.AVERAGE)
        assert isinstance(decision, EnsembleDecision)
        assert decision.composite == pytest.approx(0.85, abs=0.01)
        assert decision.passed is True
        assert decision.dissent_detected is False

    def test_strict_minimum_strategy(self):
        j1 = _mock_judge(0.90)
        j2 = _mock_judge(0.60)  # below pass threshold of 0.70
        ensemble = EnsembleJudge([("j1", j1, 0.5), ("j2", j2, 0.5)], pass_threshold=0.70)

        decision = ensemble.evaluate("input", "output", strategy=ConsensusStrategy.STRICT_MINIMUM)
        assert decision.composite == pytest.approx(0.60, abs=0.01)
        assert decision.passed is False  # conservative banking policy blocked it

    def test_weighted_strategy(self):
        j1 = _mock_judge(0.90)
        j2 = _mock_judge(0.50)
        # 0.8 weight on j1, 0.2 weight on j2 -> 0.90 * 0.8 + 0.50 * 0.2 = 0.72 + 0.10 = 0.82
        ensemble = EnsembleJudge([("j1", j1, 0.8), ("j2", j2, 0.2)])

        decision = ensemble.evaluate("input", "output", strategy=ConsensusStrategy.WEIGHTED)
        assert decision.composite == pytest.approx(0.82, abs=0.01)
        assert decision.passed is True

    def test_majority_vote_strategy(self):
        j1 = _mock_judge(0.85)  # PASS
        j2 = _mock_judge(0.80)  # PASS
        j3 = _mock_judge(0.40)  # FAIL
        ensemble = EnsembleJudge([("j1", j1, 1.0), ("j2", j2, 1.0), ("j3", j3, 1.0)], pass_threshold=0.70)

        decision = ensemble.evaluate("input", "output", strategy=ConsensusStrategy.MAJORITY_VOTE)
        assert decision.passed is True  # 2 vs 1 vote passes


class TestDissentDetection:
    def test_divergence_flagged(self):
        j1 = _mock_judge(0.95)
        j2 = _mock_judge(0.60)
        ensemble = EnsembleJudge([("j1", j1, 0.5), ("j2", j2, 0.5)], divergence_threshold=0.25)

        decision = ensemble.evaluate("input", "output")
        assert decision.divergence == pytest.approx(0.35, abs=0.01)
        assert decision.dissent_detected is True
        assert "Significant judge divergence" in decision.summary_rationale

    def test_hallucination_conflict_flagged(self):
        j1 = _mock_judge(0.90, risk="NONE")
        j2 = _mock_judge(0.60, risk="HIGH")
        ensemble = EnsembleJudge([("j1", j1, 0.5), ("j2", j2, 0.5)])

        decision = ensemble.evaluate("input", "output")
        assert decision.hallucination_conflict is True


class TestEnsembleRobustness:
    def test_empty_judges_raises(self):
        with pytest.raises(ValueError, match="at least one judge"):
            EnsembleJudge([])

    def test_judge_crash_graceful_fallback(self):
        j1 = _mock_judge(0.85)
        j2 = JudgeAgent(api_key="fake")
        j2.evaluate = MagicMock(side_effect=RuntimeError("Groq API 503 Overloaded"))

        ensemble = EnsembleJudge([("j1", j1, 0.5), ("j2", j2, 0.5)])
        decision = ensemble.evaluate("input", "output", local_fallback_score=0.75)

        assert "j1" in decision.judge_reports
        assert "j2" in decision.judge_reports
        assert decision.judge_reports["j2"].escalation_failed is True
        assert decision.composite > 0.0
