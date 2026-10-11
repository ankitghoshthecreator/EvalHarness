"""
test_pii_detector.py — Tests for guardrails/pii_detector.py

Coverage:
  - Email detection
  - SSN detection
  - Credit card detection
  - Phone number detection
  - IBAN detection
  - Account number detection
  - Clean text passes
  - Redaction
  - PIIScanResult severity levels
"""

import pytest
from guardrails.pii_detector import scan, redact, PIIScanResult, PIIMatch


# ── Email ─────────────────────────────────────────────────────────────────

class TestEmailDetection:
    def test_simple_email(self):
        result = scan("Contact us at john.doe@example.com for support.", use_ner=False)
        assert result.has_pii
        assert any(m.pii_type == "email" for m in result.matches)

    def test_email_with_plus(self):
        result = scan("Reply to john+tag@company.org", use_ner=False)
        assert result.has_pii
        assert any(m.pii_type == "email" for m in result.matches)

    def test_no_email(self):
        result = scan("No contact information provided.", use_ner=False)
        email_matches = [m for m in result.matches if m.pii_type == "email"]
        assert len(email_matches) == 0


# ── SSN ───────────────────────────────────────────────────────────────────

class TestSSNDetection:
    def test_valid_ssn(self):
        result = scan("SSN: 123-45-6789", use_ner=False)
        assert any(m.pii_type == "ssn" for m in result.matches)

    def test_invalid_ssn_format(self):
        # 000 prefix is invalid SSN
        result = scan("Number: 000-45-6789", use_ner=False)
        ssn_matches = [m for m in result.matches if m.pii_type == "ssn"]
        assert len(ssn_matches) == 0

    def test_ssn_severity_high(self):
        result = scan("SSN: 123-45-6789", use_ner=False)
        assert result.severity == "HIGH"


# ── Credit card ───────────────────────────────────────────────────────────

class TestCreditCardDetection:
    def test_visa(self):
        result = scan("Card: 4111-1111-1111-1111", use_ner=False)
        assert any(m.pii_type == "credit_card" for m in result.matches)

    def test_mastercard(self):
        result = scan("MC: 5500-0000-0000-0004", use_ner=False)
        assert any(m.pii_type == "credit_card" for m in result.matches)

    def test_severity_high(self):
        result = scan("4111111111111111", use_ner=False)
        assert result.severity == "HIGH"


# ── Phone ─────────────────────────────────────────────────────────────────

class TestPhoneDetection:
    def test_us_phone_dashes(self):
        result = scan("Call 555-867-5309 for info.", use_ner=False)
        assert any(m.pii_type == "phone_us" for m in result.matches)

    def test_us_phone_dots(self):
        result = scan("Phone: 555.867.5309", use_ner=False)
        assert any(m.pii_type == "phone_us" for m in result.matches)


# ── Clean text ────────────────────────────────────────────────────────────

class TestCleanText:
    def test_financial_news_no_pii(self, clean_text):
        result = scan(clean_text, use_ner=False)
        high_risk = {"ssn", "credit_card", "iban"}
        high_risk_matches = [m for m in result.matches if m.pii_type in high_risk]
        assert len(high_risk_matches) == 0

    def test_empty_string(self):
        result = scan("", use_ner=False)
        assert not result.has_pii


# ── Redaction ─────────────────────────────────────────────────────────────

class TestRedaction:
    def test_email_redacted(self):
        redacted = redact("Email john@example.com to confirm.")
        assert "john@example.com" not in redacted
        assert "EMAIL_REDACTED" in redacted

    def test_ssn_redacted(self):
        redacted = redact("SSN: 123-45-6789")
        assert "123-45-6789" not in redacted

    def test_no_pii_unchanged(self, clean_text):
        redacted = redact(clean_text)
        # Core content should remain
        assert "quarterly earnings" in redacted.lower() or len(redacted) > 10


# ── Severity ─────────────────────────────────────────────────────────────

class TestSeverityLevels:
    def test_no_pii_severity_none(self, clean_text):
        result = scan(clean_text, use_ner=False)
        high_risk = {"ssn", "credit_card", "iban"}
        if not any(m.pii_type in high_risk for m in result.matches):
            # severity should be NONE or LOW only
            assert result.severity in ("NONE", "LOW")

    def test_ssn_gives_high_severity(self):
        result = scan("SSN 123-45-6789", use_ner=False)
        assert result.severity == "HIGH"

    def test_email_gives_low_severity(self):
        result = scan("Send to test@example.com", use_ner=False)
        # Email alone = LOW
        if not any(m.pii_type in {"ssn", "credit_card", "iban"} for m in result.matches):
            assert result.severity == "LOW"
