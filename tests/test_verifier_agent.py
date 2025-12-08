import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from debunk_sft.utils.dataset.math.diversify_math import (
    Diversification,
    Verification,
    VerifierAgent,
)


class _NullClient:
    """Placeholder client so VerifierAgent.__init__ succeeds without OpenAI."""


def _make_agent(monkeypatch, passes: int = 3) -> VerifierAgent:
    monkeypatch.setattr(
        "debunk_sft.utils.dataset.math.diversify_math._ensure_openai_client",
        lambda: _NullClient(),
    )
    return VerifierAgent(model="dummy-model", passes=passes)


def _make_diversification() -> Diversification:
    return Diversification(
        diversified_problem="Simplify 2 + 2.",
        diversified_solution="Final answer: 4",
    )


def test_verifier_agent_majority_pass(monkeypatch):
    agent = _make_agent(monkeypatch, passes=3)

    responses = iter(
        [
            (
                "pass-raw-1",
                Verification(verdict="pass", reason="Pass reasoning A", consistency_checks={}),
            ),
            (
                "fail-raw-2",
                Verification(verdict="fail", reason="Fail reasoning", consistency_checks={}),
            ),
            (
                "pass-raw-3",
                Verification(verdict="pass", reason="Pass reasoning B", consistency_checks={}),
            ),
        ]
    )

    def fake_single_pass(*args, **kwargs):
        try:
            return next(responses)
        except StopIteration:  # pragma: no cover - defensive
            pytest.fail("VerifierAgent called more passes than expected")

    monkeypatch.setattr(agent, "_single_pass_verify", fake_single_pass)

    raw, verification = agent.verify(
        original_problem="orig problem",
        original_solution="orig solution",
        diversification=_make_diversification(),
    )

    assert "Verifier pass 1" in raw
    assert verification.verdict == "pass"
    assert (
        verification.consistency_checks["vote_summary"]["correct_votes"] == 2
    ), "Majority should count correct votes"
    assert (
        "Pass reasoning A" in verification.reason or "Pass reasoning B" in verification.reason
    ), "Majority reason should include representative reasoning from a pass vote"


def test_verifier_agent_majority_fail(monkeypatch):
    agent = _make_agent(monkeypatch, passes=3)

    responses = iter(
        [
            (
                "fail-raw-1",
                Verification(verdict="fail", reason="Fail reasoning A", consistency_checks={}),
            ),
            (
                "pass-raw-2",
                Verification(verdict="pass", reason="Pass reasoning", consistency_checks={}),
            ),
            (
                "fail-raw-3",
                Verification(verdict="fail", reason="Fail reasoning B", consistency_checks={}),
            ),
        ]
    )

    def fake_single_pass(*args, **kwargs):
        try:
            return next(responses)
        except StopIteration:  # pragma: no cover - defensive
            pytest.fail("VerifierAgent called more passes than expected")

    monkeypatch.setattr(agent, "_single_pass_verify", fake_single_pass)

    _, verification = agent.verify(
        original_problem="orig problem",
        original_solution="orig solution",
        diversification=_make_diversification(),
    )

    assert verification.verdict == "fail"
    summary = verification.consistency_checks["vote_summary"]
    assert summary["incorrect_votes"] == 2
    assert "strict majority" in summary["decision_rule"]
    assert (
        "Fail reasoning" in verification.reason
    ), "Representative reasoning should come from failing majority"

