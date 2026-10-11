"""
judge_agent.py — Groq 70B LLM-as-Judge with structured Pydantic output.

Rubric axes:
  - fluency          [0.0, 1.0]
  - factual_grounding [0.0, 1.0]
  - instruction_adherence [0.0, 1.0]
  - hallucination_risk  NONE | LOW | HIGH

Extracted and generalized from CIRCLE's GroqCriticAgent + FailureModeParser.
Includes <think> CoT-tag sanitization and fallback to local scorer on API failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema for judge output
# ---------------------------------------------------------------------------

class JudgeReport(BaseModel):
    fluency: float = Field(..., ge=0.0, le=1.0)
    factual_grounding: float = Field(..., ge=0.0, le=1.0)
    instruction_adherence: float = Field(..., ge=0.0, le=1.0)
    hallucination_risk: Literal["NONE", "LOW", "HIGH"]
    rationale: str = ""
    composite: float = 0.0
    escalation_failed: bool = False

    @field_validator("composite", mode="before")
    @classmethod
    def _set_composite(cls, v: float) -> float:
        # composite calculated after all fields are set; placeholder
        return v

    def model_post_init(self, __context):  # noqa: ANN001
        risk_penalty = {"NONE": 0.0, "LOW": 0.1, "HIGH": 0.4}[self.hallucination_risk]
        self.composite = round(
            (self.fluency + self.factual_grounding + self.instruction_adherence) / 3.0
            - risk_penalty,
            4,
        )
        self.composite = max(0.0, min(1.0, self.composite))


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a strict LLM output evaluator. Given an input and an LLM-generated output,
score the output on the following axes and return ONLY valid JSON — no markdown fences,
no extra text, no <think> tags.

JSON schema:
{
  "fluency": <float 0.0–1.0>,
  "factual_grounding": <float 0.0–1.0>,
  "instruction_adherence": <float 0.0–1.0>,
  "hallucination_risk": <"NONE" | "LOW" | "HIGH">,
  "rationale": <string, 1–3 sentences>
}

Scoring guide:
  fluency: grammar, readability, natural flow
  factual_grounding: claims are verifiable / grounded in the input
  instruction_adherence: follows the user's instruction format and constraints
  hallucination_risk: NONE = fully grounded, LOW = minor extrapolation, HIGH = fabrication
"""

_JUDGE_USER = """\
INPUT:
{input_text}

LLM OUTPUT:
{output_text}

Return only the JSON object.
"""


# ---------------------------------------------------------------------------
# Sanitise <think>...</think> CoT blocks that some models emit
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Judge agent
# ---------------------------------------------------------------------------

class JudgeAgent:
    """
    Wraps Groq 70B for structured LLM-as-judge evaluation.

    Usage:
        agent = JudgeAgent()
        report = agent.evaluate(input_text="...", output_text="...")
    """

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    TIMEOUT = 30

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.model = model or os.environ.get("GROQ_JUDGE_MODEL", self.DEFAULT_MODEL)
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
            self._available = True
        except Exception as exc:
            logger.warning("Groq client init failed: %s — judge unavailable", exc)
            self._available = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=False,
    )
    def _call_groq(self, input_text: str, output_text: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": _JUDGE_USER.format(
                        input_text=input_text, output_text=output_text
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=512,
            timeout=self.TIMEOUT,
        )
        return response.choices[0].message.content or ""

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        local_fallback_score: float | None = None,
    ) -> JudgeReport:
        """
        Evaluate (input, output) with Groq 70B.
        Falls back gracefully if API unavailable.
        """
        if not self._available:
            return self._fallback(local_fallback_score)

        try:
            raw = self._call_groq(input_text, output_text)
            clean = _strip_think_tags(raw)
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?", "", clean).replace("```", "").strip()
            data = json.loads(clean)
            return JudgeReport(**data)
        except Exception as exc:
            logger.warning("JudgeAgent evaluation failed: %s", exc)
            return self._fallback(local_fallback_score)

    @staticmethod
    def _fallback(local_score: float | None) -> JudgeReport:
        score = local_score if local_score is not None else 0.5
        return JudgeReport(
            fluency=score,
            factual_grounding=score,
            instruction_adherence=score,
            hallucination_risk="LOW",
            rationale="Groq API unavailable — using local scorer fallback.",
            escalation_failed=True,
        )
