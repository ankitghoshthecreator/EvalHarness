"""
test_agreement_metrics.py — Unit tests for statistical agreement metrics (Cohen's Kappa & Krippendorff's Alpha).
"""

import pytest
from scorer.agreement_metrics import (
    cohens_kappa,
    evaluate_agreement,
    continuous_score_agreement,
    krippendorffs_alpha_nominal,
    AgreementReport,
)


class TestCohensKappa:
    def test_perfect_agreement(self):
        r1 = ["PASS", "FAIL", "PASS", "PASS", "FAIL"]
        r2 = ["PASS", "FAIL", "PASS", "PASS", "FAIL"]
        kappa = cohens_kappa(r1, r2)
        assert kappa == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            cohens_kappa(["PASS"], ["PASS", "FAIL"])

    def test_empty_returns_one(self):
        assert cohens_kappa([], []) == 1.0

    def test_complete_disagreement(self):
        r1 = ["PASS", "PASS", "PASS", "PASS"]
        r2 = ["FAIL", "FAIL", "FAIL", "FAIL"]
        kappa = cohens_kappa(r1, r2)
        assert kappa <= 0.0

    def test_substantial_agreement_circle_benchmark(self):
        # 100 test items with ~90% agreement, reflecting human-judge alignment (κ ≈ 0.81)
        # 75 PASS-PASS, 15 FAIL-FAIL, 5 PASS-FAIL, 5 FAIL-PASS
        r1 = ["PASS"] * 80 + ["FAIL"] * 20
        r2 = ["PASS"] * 75 + ["FAIL"] * 5 + ["PASS"] * 5 + ["FAIL"] * 15
        kappa = cohens_kappa(r1, r2)
        assert 0.65 <= kappa <= 0.85

    def test_multi_category(self):
        r1 = ["LOW", "MEDIUM", "HIGH", "LOW", "HIGH"]
        r2 = ["LOW", "MEDIUM", "HIGH", "LOW", "MEDIUM"]
        kappa = cohens_kappa(r1, r2)
        assert 0.5 <= kappa <= 1.0


class TestEvaluateAgreement:
    def test_report_structure(self):
        r1 = ["PASS", "PASS", "FAIL", "PASS"]
        r2 = ["PASS", "PASS", "PASS", "PASS"]
        report = evaluate_agreement(r1, r2)
        assert isinstance(report, AgreementReport)
        assert report.total_evaluations == 4
        assert report.observed_agreement == 0.75
        assert isinstance(report.cohens_kappa, float)
        assert report.interpretation in [
            "Poor Agreement (Worse than chance)",
            "Slight Agreement",
            "Fair Agreement",
            "Moderate Agreement",
            "Substantial Agreement",
            "Almost Perfect Agreement",
        ]
        assert "PASS" in report.contingency_matrix
        assert "FAIL" in report.contingency_matrix


class TestContinuousScoreAgreement:
    def test_binarization_at_threshold(self):
        scores_judge = [0.95, 0.85, 0.40, 0.30, 0.75]
        scores_human = [0.90, 0.80, 0.50, 0.20, 0.70]
        report = continuous_score_agreement(scores_judge, scores_human, threshold=0.70)
        assert report.observed_agreement == 1.0
        assert report.cohens_kappa == 1.0

    def test_divergence_detected(self):
        scores_judge = [0.90, 0.90, 0.90]
        scores_human = [0.30, 0.20, 0.40]
        report = continuous_score_agreement(scores_judge, scores_human, threshold=0.70)
        assert report.observed_agreement == 0.0
        assert report.cohens_kappa <= 0.0


class TestKrippendorffsAlpha:
    def test_perfect_consensus_three_raters(self):
        # 4 items rated by 3 judges
        matrix = [
            ["PASS", "PASS", "PASS"],
            ["FAIL", "FAIL", "FAIL"],
            ["PASS", "PASS", "PASS"],
            ["FAIL", "FAIL", "FAIL"],
        ]
        alpha = krippendorffs_alpha_nominal(matrix)
        assert alpha == 1.0

    def test_missing_data_handled(self):
        # Some items only evaluated by 2 of 3 judges
        matrix = [
            ["PASS", "PASS", None],
            ["FAIL", "FAIL", "FAIL"],
            [None, "PASS", "PASS"],
            ["FAIL", None, "FAIL"],
        ]
        alpha = krippendorffs_alpha_nominal(matrix)
        assert alpha == 1.0

    def test_disagreement_lowers_alpha(self):
        matrix = [
            ["PASS", "FAIL", "PASS"],
            ["FAIL", "PASS", "FAIL"],
            ["PASS", "PASS", "FAIL"],
        ]
        alpha = krippendorffs_alpha_nominal(matrix)
        assert alpha < 0.5
