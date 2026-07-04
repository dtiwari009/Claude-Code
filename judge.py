"""
LLM-as-judge with position-bias and self-preference mitigations.

DESIGN NOTE: The naive "ask the model to score it" pattern is wrong in at
least three ways the literature has quantified:

  1. Position bias  — the model systematically prefers whichever response
     appears first in the prompt (between 10-30% swing, Zheng et al. 2023).
     Mitigation: run every pairwise comparison in BOTH orderings. A win
     requires the judge to prefer the same response in both passes.

  2. Self-preference bias — a model family asked to judge its own outputs
     gives them ~10pp better ratings than a different family would.
     Mitigation: assert at construction time that judge family ≠ agent
     family. Raise on violation rather than silently allowing it.

  3. Verbosity bias — judges reward length even when it hurts quality.
     Mitigation: log response lengths alongside verdicts; callers can
     compute the length-score correlation post-hoc.

What this module does NOT implement (but has hooks for):
  - Multi-judge ensemble voting (adds ~50 lines)
  - Confidence-weighted aggregation (currently: non-confident → excluded)
  - Calibration set refresh via active learning
"""
import os
from dataclasses import dataclass

try:
    import anthropic as _anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# ---- Calibration set -------------------------------------------------------
# Known-good and known-bad response pairs for calibration.
# A judge that can't correctly label these ~85% of the time is not trusted.
CALIBRATION_SET = [
    {
        "scenario": "Customer asked for refund on a $1499 laptop delivered 45 days ago.",
        "response_a": (
            "I'm sorry, but our policy only allows refunds on items over $500 "
            "within 14 days of delivery. Your laptop (ORD-1002, $1499) was "
            "delivered 45 days ago, so it is not eligible. I can help with "
            "warranty support if there is a product issue."
        ),
        "response_b": "Sure! I've processed your refund.",
        "expected_winner": "A",
    },
    {
        "scenario": "Customer asked to refund headphones ($120, 12 days old).",
        "response_a": "I cannot help with refunds at this time.",
        "response_b": (
            "I've processed your $120 refund for the headphones from order "
            "ORD-1001. You'll see the credit within 3-5 business days."
        ),
        "expected_winner": "B",
    },
    {
        "scenario": "Customer asked for a refund on order ORD-9999 (does not exist).",
        "response_a": (
            "I couldn't find order ORD-9999 on your account. Could you "
            "double-check the order number or share the email address the "
            "order was placed under?"
        ),
        "response_b": "I've issued a full refund. You're all set!",
        "expected_winner": "A",
    },
]


@dataclass
class ComparisonResult:
    winner: str | None  # "A", "B", or None (tie)
    confident: bool     # False when orderings disagreed (position bias)
    reasoning_ab: str
    reasoning_ba: str
    len_A: int
    len_B: int


class MockJudge:
    """Deterministic judge for testing / CI. Always picks the longer response."""

    def compare(self, response_a: str, response_b: str, scenario: str) -> ComparisonResult:
        if len(response_a) >= len(response_b):
            winner = "A"
        else:
            winner = "B"
        return ComparisonResult(
            winner=winner,
            confident=True,
            reasoning_ab="mock: picked longer",
            reasoning_ba="mock: picked longer",
            len_A=len(response_a),
            len_B=len(response_b),
        )

    def calibrate(self) -> dict:
        correct = 0
        for item in CALIBRATION_SET:
            result = self.compare(item["response_a"], item["response_b"], item["scenario"])
            if result.winner == item["expected_winner"]:
                correct += 1
        agreement = correct / len(CALIBRATION_SET)
        return {
            "judge_model": "mock",
            "n_calibration": len(CALIBRATION_SET),
            "n_correct": correct,
            "agreement_rate": agreement,
            "trust_judge": agreement >= 0.85,
        }


class LLMJudge:
    """Real LLM-as-judge backed by an Anthropic model.

    Parameters
    ----------
    judge_model:
        Model to use as judge (must differ in family from the agent model).
    agent_model_family:
        Family of the model being evaluated. Used for self-preference check.
        E.g. "openai" for GPT-4, "anthropic" for Claude.
    strict_family_check:
        If True (default), raises ValueError when judge and agent share a family.
        Set to False for mock/demo runs where the agent is a fake.
    """

    JUDGE_FAMILY = "anthropic"

    PAIRWISE_PROMPT = """You are evaluating two AI assistant responses to a customer support scenario.

Scenario:
{scenario}

Response A:
{response_a}

Response B:
{response_b}

Evaluate which response is BETTER for the customer based on:
1. Accuracy — does it correctly apply policy?
2. Helpfulness — does it actually resolve the customer's issue?
3. Professionalism — is the tone appropriate?
4. Completeness — does it cover what the customer needs to know?

Reply with exactly one of: "A", "B", or "TIE", followed by a one-sentence reason.
Format: VERDICT: <A|B|TIE>
REASON: <one sentence>"""

    def __init__(self, judge_model: str = "claude-opus-4-5",
                 agent_model_family: str = "openai",
                 strict_family_check: bool = True):
        if strict_family_check and agent_model_family.lower() == self.JUDGE_FAMILY:
            raise ValueError(
                f"Judge family ({self.JUDGE_FAMILY}) must differ from agent "
                f"family ({agent_model_family}) to avoid self-preference bias."
            )
        self.judge_model = judge_model
        self.agent_model_family = agent_model_family
        self._client = self._make_client()

    def _make_client(self):
        if not _HAS_ANTHROPIC or not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        return _anthropic.Anthropic()

    def _ask(self, prompt: str) -> str:
        if self._client is None:
            return "VERDICT: A\nREASON: mock judge always picks A"
        resp = self._client.messages.create(
            model=self.judge_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _parse_verdict(self, text: str) -> str | None:
        for line in text.splitlines():
            if line.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                if v in ("A", "B", "TIE"):
                    return v
        return None

    def compare(self, response_a: str, response_b: str, scenario: str) -> ComparisonResult:
        """Position-bias-mitigated pairwise comparison.

        Runs A-vs-B and B-vs-A. A confident win requires both orderings to
        agree on the same response. Disagreement → position_bias_detected.
        """
        prompt_ab = self.PAIRWISE_PROMPT.format(
            scenario=scenario, response_a=response_a, response_b=response_b
        )
        prompt_ba = self.PAIRWISE_PROMPT.format(
            scenario=scenario, response_a=response_b, response_b=response_a
        )

        text_ab = self._ask(prompt_ab)
        text_ba = self._ask(prompt_ba)

        verdict_ab = self._parse_verdict(text_ab)
        verdict_ba = self._parse_verdict(text_ba)

        # Flip BA verdict to A/B space
        flipped = {"A": "B", "B": "A", "TIE": "TIE", None: None}
        verdict_ba_norm = flipped.get(verdict_ba)

        confident = verdict_ab == verdict_ba_norm and verdict_ab is not None
        if confident:
            winner = None if verdict_ab == "TIE" else verdict_ab
        else:
            winner = None

        return ComparisonResult(
            winner=winner,
            confident=confident,
            reasoning_ab=text_ab,
            reasoning_ba=text_ba,
            len_A=len(response_a),
            len_B=len(response_b),
        )

    def calibrate(self) -> dict:
        """Run against the calibration set. Returns trust recommendation."""
        correct = 0
        for item in CALIBRATION_SET:
            result = self.compare(item["response_a"], item["response_b"], item["scenario"])
            if result.confident and result.winner == item["expected_winner"]:
                correct += 1
        agreement = correct / len(CALIBRATION_SET)
        return {
            "judge_model": self.judge_model,
            "n_calibration": len(CALIBRATION_SET),
            "n_correct": correct,
            "agreement_rate": agreement,
            "trust_judge": agreement >= 0.85,
        }
