"""
    Coverage & Exclusion Agent: assesses coverage scope, waiting periods,
    exclusions, and limits using only the retrieved policy evidence.
"""
import logging

from src.config import Settings
from src.agents.base_agent import BaseLLMAgent
from src.models.schemas import CoverageFindings


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert Coverage & Exclusion Agent. Assess coverage scope, "
    "waiting periods, exclusions, definitions, and applicable limits using "
    "ONLY the provided policy evidence. If the evidence is insufficient to "
    "make a safe conclusion, set needs_review to True. Do not use external "
    "insurance knowledge."
)
HUMAN_PROMPT = (
    "Extracted Claim Facts:\n{facts}\n\nRetrieved Policy Evidence:\n{evidence}\n\nAssess the coverage."
)


class CoverageAgent(BaseLLMAgent):
    def __init__(self, settings: Settings):
        super().__init__(
            settings = settings,
            output_schema = CoverageFindings,
            system_prompt = SYSTEM_PROMPT,
            human_prompt = HUMAN_PROMPT,
            agent_name = "Coverage & Exclusion",
        )


    def invoke(self, state: dict) -> dict:
        """
            Evaluates extracted facts against the retrieved evidence to determine coverage.
        """
        case_id = state.get("case_id", "UNKNOWN")
        logger.info("Assessing coverage for case: %s", case_id)

        facts = state.get("extracted_facts", {})
        evidence = state.get("retrieved_evidence", [])

        if not evidence:
            logger.warning("No evidence provided to Coverage Agent.")
            fallback = CoverageFindings(
                is_covered = False,
                reasoning = "No evidence retrieved to assess coverage.",
                needs_review = True,
            )
            return {
                "coverage_findings": fallback.model_dump(),
                "trace": [self.trace_entry("Failed due to missing evidence")],
            }

        # Format evidence for the prompt so the LLM can see the Chunk IDs
        formatted_evidence = "\n\n".join(
            f"Chunk ID: {doc.metadata.get('chunk_id')}\n{doc.page_content}" for doc in evidence
        )

        logger.info("Invoking LLM for coverage assessment...")
        result = self._safe_invoke({"facts": str(facts), "evidence": formatted_evidence})

        if result is None:
            fallback = CoverageFindings(reasoning = "Coverage assessment failed due to an LLM error; flagged for manual review.", needs_review = True)
            return {
                "coverage_findings": fallback.model_dump(),
                "trace": [self.trace_entry("Failed: LLM invocation error during coverage assessment")],
            }

        return {
            "coverage_findings": result.model_dump(),
            "trace": [self.trace_entry("Evaluated facts against retrieved evidence")],
        }


if __name__ == "__main__":
    from src.config import configure_logging, load_settings
    from src.ingestion.chunking import ingest_policy
    from src.retrieval.hybrid_search import HybridRetriever
    from src.agents.policy_evidence import PolicyEvidenceAgent

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
            "Verify policy is active as of the claim date",
            "Review room rent capping and limits",
            "Check for any specific waiting periods for Appendectomy",
        ],
    }

    # 3. Retrieve Evidence
    evidence_state = evidence_agent.invoke(state = test_state)
    test_state.update(evidence_state)

    # 4. Run the Coverage Agent
    coverage_agent = CoverageAgent(settings = settings)
    output_state = coverage_agent.invoke(state = test_state)

    logger.info("Coverage Findings: %s", output_state["coverage_findings"])
    logger.info("Trace: %s", output_state["trace"])
