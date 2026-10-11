"""
agreement_metrics.py — Statistical Inter-Annotator Agreement Engine (Pure Python).

Implements:
  - Cohen's Kappa (κ): standard rater-to-rater agreement corrected for chance
  - Krippendorff's Alpha (α): multi-judge consensus supporting arbitrary raters
  - Continuous-to-categorical thresholding & confusion matrix analysis
  - Landis & Koch (1977) qualitative benchmark interpretation

Directly underpins the CIRCLE human-judge alignment benchmark (κ = 0.81).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class AgreementReport:
    observed_agreement: float      # Po
    chance_agreement: float        # Pe
    cohens_kappa: float            # κ
    interpretation: str            # e.g., "Substantial Agreement"
    total_evaluations: int
    contingency_matrix: dict[str, dict[str, int]]


def _interpret_kappa(kappa: float) -> str:
    """Landis and Koch (1977) scale for interpreting kappa coefficients."""
    if kappa < 0.0:
        return "Poor Agreement (Worse than chance)"
    elif kappa <= 0.20:
        return "Slight Agreement"
    elif kappa <= 0.40:
        return "Fair Agreement"
    elif kappa <= 0.60:
        return "Moderate Agreement"
    elif kappa <= 0.80:
        return "Substantial Agreement"
    else:
        return "Almost Perfect Agreement"


def cohens_kappa(rater1: Sequence[Any], rater2: Sequence[Any]) -> float:
    """
    Calculate Cohen's Kappa for two raters on categorical labels.

    Args:
        rater1: List of ratings from rater 1
        rater2: List of ratings from rater 2

    Returns:
        float κ ∈ [-1, 1], where 1 = perfect agreement, 0 = chance agreement
    """
    if len(rater1) != len(rater2):
        raise ValueError(f"Rating length mismatch: {len(rater1)} vs {len(rater2)}")
    n = len(rater1)
    if n == 0:
        return 1.0

    categories = sorted(set(rater1) | set(rater2))
    if len(categories) <= 1:
        return 1.0

    # Observed agreement
    agreements = sum(1 for a, b in zip(rater1, rater2) if a == b)
    p_o = agreements / n

    # Expected agreement by chance
    count1 = Counter(rater1)
    count2 = Counter(rater2)
    p_e = sum((count1[cat] / n) * (count2[cat] / n) for cat in categories)

    if p_e >= 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(kappa, 4)


def evaluate_agreement(
    ratings1: Sequence[Any],
    ratings2: Sequence[Any],
) -> AgreementReport:
    """
    Compute complete statistical agreement report with contingency matrix.
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("Ratings must have equal length")
    n = len(ratings1)
    categories = sorted(set(ratings1) | set(ratings2))

    # Contingency matrix
    matrix: dict[str, dict[str, int]] = {
        str(c1): {str(c2): 0 for c2 in categories} for c1 in categories
    }
    for r1, r2 in zip(ratings1, rater2_safe := ratings2):
        matrix[str(r1)][str(r2)] += 1

    agreements = sum(1 for a, b in zip(ratings1, ratings2) if a == b)
    p_o = round(agreements / n, 4) if n > 0 else 1.0

    count1 = Counter(ratings1)
    count2 = Counter(ratings2)
    p_e = round(sum((count1[cat] / n) * (count2[cat] / n) for cat in categories), 4) if n > 0 else 1.0

    kappa = cohens_kappa(ratings1, ratings2)
    return AgreementReport(
        observed_agreement=p_o,
        chance_agreement=p_e,
        cohens_kappa=kappa,
        interpretation=_interpret_kappa(kappa),
        total_evaluations=n,
        contingency_matrix=matrix,
    )


def continuous_score_agreement(
    scores1: Sequence[float],
    scores2: Sequence[float],
    threshold: float = 0.70,
) -> AgreementReport:
    """
    Binarize continuous scores (>= threshold is PASS, else FAIL) and compute agreement.
    """
    labels1 = ["PASS" if s >= threshold else "FAIL" for s in scores1]
    labels2 = ["PASS" if s >= threshold else "FAIL" for s in scores2]
    return evaluate_agreement(labels1, labels2)


def krippendorffs_alpha_nominal(matrix: Sequence[Sequence[Any | None]]) -> float:
    """
    Compute Krippendorff's Alpha (nominal metric) across multiple raters.

    matrix: list of units (items), each unit is a list of ratings from R raters.
            None denotes missing observations.
    """
    units = matrix
    if not units:
        return 1.0

    # Collect all unique values
    values = sorted({val for u in units for val in u if val is not None})
    if len(values) <= 1:
        return 1.0

    val_to_idx = {v: i for i, v in enumerate(values)}
    num_vals = len(values)

    # Coincidence matrix
    coincidence = [[0.0] * num_vals for _ in range(num_vals)]
    total_pairable = 0.0

    for unit in units:
        valid_ratings = [r for r in unit if r is not None]
        m = len(valid_ratings)
        if m < 2:
            continue  # cannot compare single rating

        norm = m - 1.0
        counts = Counter(valid_ratings)
        for v1, c1 in counts.items():
            i1 = val_to_idx[v1]
            for v2, c2 in counts.items():
                i2 = val_to_idx[v2]
                pairs = (c1 * (c1 - 1) if i1 == i2 else c1 * c2) / norm
                coincidence[i1][i2] += pairs
                total_pairable += pairs

    if total_pairable == 0.0:
        return 1.0

    # Marginals
    n_v = [sum(coincidence[i]) for i in range(num_vals)]
    n_total = sum(n_v)

    # Observed disagreement
    d_o = sum(
        coincidence[i][j]
        for i in range(num_vals)
        for j in range(num_vals)
        if i != j
    ) / n_total

    # Expected disagreement
    d_e = sum(
        n_v[i] * n_v[j]
        for i in range(num_vals)
        for j in range(num_vals)
        if i != j
    ) / (n_total * (n_total - 1.0)) if n_total > 1.0 else 1.0

    if d_e == 0.0:
        return 1.0

    alpha = 1.0 - (d_o / d_e)
    return round(alpha, 4)
