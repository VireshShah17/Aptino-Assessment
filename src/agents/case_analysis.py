"""
    Case Analysis Agent: extracts structured facts and an investigation plan
    from raw claim data.
"""
import logging

from src.config import Settings
from src.agents.base_agent import BaseLLMAgent
from src.models.schemas import AnalysisOutput


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert health insurance Case Analysis Agent. Your job is to "
    "extract facts, identify decision dimensions, detect missing fields, and "
    "create a strict investigation plan. Do not make a final decision. Only "
    "output the structured extraction."
)
HUMAN_PROMPT = "Analyze the following raw claim data and return the structured analysis:\n\n{claim_data}"


class CaseAnalysisAgent(BaseLLMAgent):
    def __init__(self, settings: Settings):
        super().__init__(
            settings = settings,
            output_schema = AnalysisOutput,
            system_prompt = SYSTEM_PROMPT,
            human_prompt = HUMAN_PROMPT,
            agent_name = "Case Analysis",
        )


    def invoke(self, state: dict) -> dict:
        """
            Processes the raw claim data and returns state updates.
        """
        case_id = state.get("case_id", "UNKNOWN")
        logger.info("Analyzing claim case: %s", case_id)

        raw_data = state.get("raw_claim_data", {})

        logger.info("Invoking LLM for structured fact extraction...")
        result = self._safe_invoke({"claim_data": str(raw_data)})

        if result is None:
            return {
                "extracted_facts": {},
                "investigation_plan": [],
                "trace": [self.trace_entry("Failed: LLM invocation error during fact extraction")],
            }

        return {
            "extracted_facts": result.extracted_facts,
            "investigation_plan": result.investigation_plan,
            "trace": [self.trace_entry("Extracted facts and generated investigation plan")],
        }


if __name__ == "__main__":
    from src.config import configure_logging, load_settings

    configure_logging()
    settings = load_settings()

    # Simulating a LangGraph state using PUB-001 from the public test cases
    test_state = {
        "case_id": "PUB-001",
        "raw_claim_data": {
            "policy_start_date": "2025-01-01",
            "claim_date": "2026-03-14",
            "treatment": {
                "diagnosis": "Acute appendicitis",
                "procedure": "Appendectomy",
                "pre_existing": False,
            },
            "expenses_inr": {
                "room": 30000,
                "doctor_fees": 30000,
            },
        },
    }

    agent = CaseAnalysisAgent(settings = settings)
    output_state = agent.invoke(state = test_state)

    logger.info("Extracted Facts: %s", output_state["extracted_facts"])
    logger.info("Investigation Plan: %s", output_state["investigation_plan"])
    logger.info("Trace: %s", output_state["trace"])
