"""
    Validation Agent: verifies that decision citations are actually grounded
    in the retrieved policy evidence.
"""
import logging

from src.config import Settings
from src.agents.base_agent import BaseLLMAgent
from src.models.schemas import ValidationResult


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a strict Validation Agent for an insurance decision engine. "
    "Your sole job is to verify if the provided 'claims/citations' are "
    "explicitly supported by the 'retrieved evidence'. If a claim is not "
    "supported by the evidence, mark status as 'FAIL' and list the "
    "unsupported claims. Otherwise, return 'PASS'."
)
HUMAN_PROMPT = "Decision Citations:\n{citations}\n\nRetrieved Evidence:\n{evidence}\n\nValidate the citations."


class ValidationAgent(BaseLLMAgent):
    def __init__(self, settings: Settings):
        super().__init__(
            settings = settings,
            output_schema = ValidationResult,
            system_prompt = SYSTEM_PROMPT,
            human_prompt = HUMAN_PROMPT,
            agent_name = "Validation",
        )


    def invoke(self, state: dict) -> dict:
        """
            Verifies that the decision citations are grounded in the retrieved evidence.
        """
        case_id = state.get("case_id", "UNKNOWN")
        logger.info("Validating citations for case: %s", case_id)

        citations = state.get("citations", [])
        evidence = state.get("retrieved_evidence", [])

        if not citations:
            logger.warning("No citations to validate.")
            return {
                "validation_status": "FAIL",
                "unsupported_claims": ["No citations provided by Decision Agent."],
                "trace": [self.trace_entry("Failed: No citations found")],
            }

        # Format evidence and citations for the prompt
        formatted_evidence = "\n\n".join(
            f"Chunk ID: {doc.metadata.get('chunk_id')}\nContent: {doc.page_content}" for doc in evidence
        )
        formatted_citations = "\n".join(
            f"Claim: {c.get('claim', '')} (Source Chunk: {c.get('chunk_id', '')})" for c in citations
        )

        logger.info("Invoking LLM for citation validation...")
        result = self._safe_invoke({"citations": formatted_citations, "evidence": formatted_evidence})

        if result is None:
            return {
                "validation_status": "FAIL",
                "unsupported_claims": ["Validation failed due to an LLM error; flagged for manual review."],
                "trace": [self.trace_entry("Failed: LLM invocation error during validation")],
            }

        return {
            "validation_status": result.status,
            "unsupported_claims": result.unsupported_claims,
            "trace": [self.trace_entry(f"Validation status: {result.status}")],
        }


if __name__ == "__main__":
    from src.config import configure_logging, load_settings
    from src.ingestion.chunking import ingest_policy
    from src.retrieval.hybrid_search import HybridRetriever
    from src.agents.policy_evidence import PolicyEvidenceAgent
    from src.agents.coverage_agent import CoverageAgent
    from src.agents.decision_agent import DecisionAgent

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

    # 3. Retrieve Evidence, Assess Coverage, and Make Decision
    test_state.update(evidence_agent.invoke(state = test_state))

    coverage_agent = CoverageAgent(settings = settings)
    test_state.update(coverage_agent.invoke(state = test_state))

    decision_agent = DecisionAgent(settings = settings)
    test_state.update(decision_agent.invoke(state = test_state))

    # 4. Run the Validation Agent
    validation_agent = ValidationAgent(settings = settings)
    output_state = validation_agent.invoke(state = test_state)

    logger.info("Validation Status: %s", output_state["validation_status"])
    logger.info("Unsupported Claims: %s", output_state["unsupported_claims"])
    logger.info("Trace: %s", output_state["trace"])
