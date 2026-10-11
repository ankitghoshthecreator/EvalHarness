"""
conftest.py — Shared fixtures and test configuration.
"""
import pytest
from pathlib import Path

@pytest.fixture
def probe_bank_path():
    return Path(__file__).parent.parent / "probes" / "probe_bank.yaml"

@pytest.fixture
def sample_input():
    return (
        "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
        "as policymakers continue efforts to reduce inflation toward their 2% target."
    )

@pytest.fixture
def sample_output():
    return (
        "The Federal Reserve increased interest rates by 25 basis points to combat "
        "ongoing inflation, marking a continued tightening of monetary policy."
    )

@pytest.fixture
def circular_output(sample_input):
    """Output that nearly mirrors the input — should be flagged as circular."""
    return sample_input  # exact copy = maximum circularity

@pytest.fixture
def repetitive_output():
    return "The market is up. The market is up. The market is up. The market is up. " * 5

@pytest.fixture
def pii_text():
    return (
        "Customer John Smith, SSN 123-45-6789, credit card 4111-1111-1111-1111, "
        "email john@example.com, called about account number 987654321012."
    )

@pytest.fixture
def clean_text():
    return "The quarterly earnings exceeded analyst expectations by 12 percent."
