"""
    Decision Agent: synthesizes facts and coverage findings into a final,
    citation-backed decision.
"""
import logging

from src.config import Settings
from src.agents.base_agent import BaseLLMAgent
from src.models.schemas import DecisionOutput, VALID_DECISIONS


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the final Decision Agent for a health insurance claim system. "
    "Synthesize the facts and coverage findings into a final decision. You "
    "MUST return one of the predefined decision statuses. Back up material "
    "claims with precise citations using the provided chunk metadata. If "
    "coverage findings indicate needs_review, the decision MUST be NEEDS_REVIEW."
)
HUMAN_PROMPT = (
    "Extracted Facts:\n{facts}\n\nCoverage Findings:\n{coverage}\n\n"
    "Retrieved Evidence:\n{evidence}\n\nFormulate the final decision."
)


class DecisionAgent(BaseLLMAgent):
    def __init__(self, settings: Settings):
        super().__init__(
            settings = settings,
            output_schema = DecisionOutput,
            system_prompt = SYSTEM_PROMPT,
            human_prompt = HUMAN_PROMPT,
            agent_name = "Decision",
        )


    def invoke(self, state: dict) -> dict:
        """
            Formulates the final decision contract based on prior agent findings.
        """
        case_id = state.get("case_id", "UNKNOWN")
        logger.info("Formulating final decision for case: %s", case_id)

        facts = state.get("extracted_facts", {})
        coverage = state.get("coverage_findings", {})
        evidence = state.get("retrieved_evidence", [])

        # Formatting evidence with explicit metadata for citation generation
        formatted_evidence = "\n\n".join(
            f"Chunk ID: {doc.metadata.get('chunk_id')}\n"
            f"Section: {doc.metadata.get('Section', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in evidence
        )

        logger.info("Invoking LLM for final decision synthesis...")
        result = self._safe_invoke(
            {"facts": str(facts), "coverage": str(coverage), "evidence": formatted_evidence}
        )

        if result is None:
            return {
                "final_decision": "NEEDS_REVIEW",
                "confidence": 0.0,
                "key_findings": [],
                "applicable_limits": [],
                "missing_evidence": ["Decision synthesis failed due to an LLM error."],
                "citations": [],
                "trace": [self.trace_entry("Failed: LLM invocation error during decision synthesis")],
            }

        decision = result.decision
        confidence = result.confidence

        # --- Hard guardrail --------------------------------------------
        if coverage.get("needs_review") and decision != "NEEDS_REVIEW":
            logger.warning(
                "Overriding LLM decision '%s' -> 'NEEDS_REVIEW' because "
                "coverage_findings.needs_review is True.",
                decision,
            )
            decision = "NEEDS_REVIEW"
            confidence = min(confidence, 0.5)

        if decision not in VALID_DECISIONS:
            logger.warning("LLM returned an unrecognized decision '%s'; forcing NEEDS_REVIEW.", decision)
            decision = "NEEDS_REVIEW"
            confidence = 0.0

        return {
            "final_decision": decision,
            "confidence": confidence,
            "key_findings": result.key_findings,
            "applicable_limits": result.applicable_limits,
            "missing_evidence": result.missing_evidence,
            "citations": [cit.model_dump() for cit in result.citations],
            "trace": [
                self.trace_entry(f"Made decision: {decision} with {len(result.citations)} citations")
            ],
        }


if __name__ == "__main__":
    from src.config import configure_logging, load_settings
    from src.ingestion.chunking import ingest_policy
    from src.retrieval.hybrid_search import HybridRetriever
    from src.agents.policy_evidence import PolicyEvidenceAgent
    from src.agents.coverage_agent import CoverageAgent

    configure_logging()
    settings = load_settings()

    # 1. Setup Retriever & Evidence Agent
    target_pdf = "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    chunks = ingest_policy(target_pdf)
    hybrid_retriever = HybridRetriever(chunks = chunks)
    evidence_agent = PolicyEvidenceAgent(retriever = hybrid_retriever)

    # 2. Simulate state after Case Analysis
    test_state = {
        "case_id": "PUB-001",
        "extracted_facts": {
            "policy_start_date": "2025-01-01",
            "claim_date": "2026-03-14",
            "diagnosis": "Acute appendicitis",
            "procedure": "Appendectomy",
            "pre_existing": False,
            "room_expense_inr": 30000,
            "doctor_fees_inr": 30000,
        },
        "investigation_plan": [
            "Review room rent capping and limits",
            "Check for any specific waiting periods for Appendectomy",
        ],
    }

    # 3. Retrieve Evidence & Assess Coverage
    test_state.update(evidence_agent.invoke(state = test_state))

    coverage_agent = CoverageAgent(settings = settings)
    test_state.update(coverage_agent.invoke(state = test_state))

    # 4. Run the Decision Agent
    decision_agent = DecisionAgent(settings = settings)
    output_state = decision_agent.invoke(state = test_state)

    logger.info("Final Decision: %s", output_state["final_decision"])
    logger.info("Confidence: %s", output_state["confidence"])
    logger.info("Citations: %s", output_state["citations"])
    logger.info("Trace: %s", output_state["trace"])
