"""
    Unit tests for the agent layer, run entirely offline with the LLM chain
    mocked out. These exercise the code-level fallback schemas and
    guardrails described in the code review, without needing a live
    GOOGLE_API_KEY or network access.

    Run with: pytest tests/test_agents.py -v
"""
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.models.schemas import CoverageFindings, DecisionOutput, Citation


@pytest.fixture
def settings():
    return Settings(google_api_key = "test-key")


@patch("src.agents.base_agent.ChatGoogleGenerativeAI")
def test_coverage_agent_no_evidence_returns_full_schema(mock_llm_cls, settings):
    """
        The 'no evidence' fallback must return every CoverageFindings field,
        not just needs_review/reasoning, so downstream agents never KeyError.
    """
    from src.agents.coverage_agent import CoverageAgent

    mock_llm_cls.return_value.with_structured_output.return_value = MagicMock()
    agent = CoverageAgent(settings = settings)

    result = agent.invoke({"case_id": "T-1", "extracted_facts": {}, "retrieved_evidence": []})

    findings = result["coverage_findings"]
    for field in CoverageFindings.model_fields:
        assert field in findings
    assert findings["needs_review"] is True


@patch("src.agents.base_agent.ChatGoogleGenerativeAI")
def test_coverage_agent_llm_failure_returns_full_schema(mock_llm_cls, settings):
    """
        An LLM exception must not propagate - it should degrade to a full,
        needs_review CoverageFindings instead of crashing the graph.
    """
    from src.agents.coverage_agent import CoverageAgent

    mock_llm_cls.return_value.with_structured_output.return_value = MagicMock()
    agent = CoverageAgent(settings = settings)
    agent.chain = MagicMock()
    agent.chain.invoke.side_effect = RuntimeError("boom")

    fake_doc = MagicMock()
    fake_doc.metadata = {"chunk_id": "c1"}
    fake_doc.page_content = "some policy text"

    result = agent.invoke({"case_id": "T-1", "extracted_facts": {}, "retrieved_evidence": [fake_doc]})

    findings = result["coverage_findings"]
    for field in CoverageFindings.model_fields:
        assert field in findings
    assert findings["needs_review"] is True


@patch("src.agents.base_agent.ChatGoogleGenerativeAI")
def test_decision_agent_forces_needs_review_when_coverage_flags_it(mock_llm_cls, settings):
    """
        Even if the LLM ignores the system-prompt instruction and returns
        ADMISSIBLE, the code-level guardrail must override it to NEEDS_REVIEW
        whenever coverage_findings.needs_review is True.
    """
    from src.agents.decision_agent import DecisionAgent

    mock_llm_cls.return_value.with_structured_output.return_value = MagicMock()
    agent = DecisionAgent(settings = settings)
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = DecisionOutput(
        decision = "ADMISSIBLE",
        confidence = 0.95,
        key_findings = ["Looks fine"],
        applicable_limits = [],
        missing_evidence = [],
        citations = [Citation(chunk_id = "c1", claim = "Covered under policy")],
    )

    state = {
        "case_id": "T-1",
        "extracted_facts": {},
        "coverage_findings": {"needs_review": True},
        "retrieved_evidence": [],
    }

    result = agent.invoke(state)

    assert result["final_decision"] == "NEEDS_REVIEW"
    assert result["confidence"] <= 0.5


@patch("src.agents.base_agent.ChatGoogleGenerativeAI")
def test_decision_agent_llm_failure_returns_needs_review(mock_llm_cls, settings):
    """
        An LLM exception during decision synthesis must degrade safely to
        NEEDS_REVIEW rather than crashing the pipeline.
    """
    from src.agents.decision_agent import DecisionAgent

    mock_llm_cls.return_value.with_structured_output.return_value = MagicMock()
    agent = DecisionAgent(settings = settings)
    agent.chain = MagicMock()
    agent.chain.invoke.side_effect = RuntimeError("boom")

    state = {
        "case_id": "T-1",
        "extracted_facts": {},
        "coverage_findings": {},
        "retrieved_evidence": [],
    }

    result = agent.invoke(state)

    assert result["final_decision"] == "NEEDS_REVIEW"
    assert result["confidence"] == 0.0


@patch("src.agents.base_agent.ChatGoogleGenerativeAI")
def test_decision_agent_rejects_unrecognized_decision(mock_llm_cls, settings):
    """
        A decision string outside VALID_DECISIONS must be forced to
        NEEDS_REVIEW rather than passed through as-is.
    """
    from src.agents.decision_agent import DecisionAgent

    mock_llm_cls.return_value.with_structured_output.return_value = MagicMock()
    agent = DecisionAgent(settings = settings)
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = DecisionOutput(
        decision = "APPROVED",  # not a valid status
        confidence=0.8,
        key_findings = [],
        applicable_limits = [],
        missing_evidence = [],
        citations = [],
    )

    state = {
        "case_id": "T-1",
        "extracted_facts": {},
        "coverage_findings": {"needs_review": False},
        "retrieved_evidence": [],
    }

    result = agent.invoke(state)
    assert result["final_decision"] == "NEEDS_REVIEW"


def test_settings_requires_api_key(monkeypatch):
    """
        load_settings() must fail fast with a clear error if GOOGLE_API_KEY
        is not set, rather than letting a downstream agent fail cryptically.
    """
    from src.config import load_settings

    monkeypatch.delenv("GOOGLE_API_KEY", raising = False)
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)

    with pytest.raises(ValueError):
        load_settings()
