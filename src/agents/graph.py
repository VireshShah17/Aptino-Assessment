"""
    Builds and runs the LangGraph state machine for the claim decision engine.
"""
import argparse
import logging

from langgraph.graph import StateGraph, END

from src.config import configure_logging, load_settings, Settings
from src.models.state import ClaimGraphState
from src.agents.case_analysis import CaseAnalysisAgent
from src.agents.policy_evidence import PolicyEvidenceAgent
from src.agents.coverage_agent import CoverageAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.validation_agent import ValidationAgent
from src.retrieval.hybrid_search import HybridRetriever
from src.ingestion.chunking import ingest_policy


logger = logging.getLogger(__name__)


def build_claim_graph(pdf_path: str, settings: Settings):
    """
        Builds and compiles the LangGraph state machine for the claim decision engine.
    """
    logger.info("Building LangGraph state machine...")

    # 1. Initialize core infrastructure
    logger.info("Initializing RAG infrastructure...")
    chunks = ingest_policy(pdf_path)
    retriever = HybridRetriever(chunks = chunks)

    # 2. Initialize Agents
    logger.info("Initializing specialized agents...")
    analysis_agent = CaseAnalysisAgent(settings = settings)
    evidence_agent = PolicyEvidenceAgent(retriever = retriever)
    coverage_agent = CoverageAgent(settings = settings)
    decision_agent = DecisionAgent(settings = settings)
    validation_agent = ValidationAgent(settings = settings)

    # 3. Define the Graph
    workflow = StateGraph(ClaimGraphState)

    # Add nodes
    workflow.add_node("case_analysis", analysis_agent.invoke)
    workflow.add_node("policy_evidence", evidence_agent.invoke)
    workflow.add_node("coverage_assessment", coverage_agent.invoke)
    workflow.add_node("decision_synthesis", decision_agent.invoke)
    workflow.add_node("validation", validation_agent.invoke)

    # Define edges (Linear flow for this pipeline)
    workflow.set_entry_point("case_analysis")
    workflow.add_edge("case_analysis", "policy_evidence")
    workflow.add_edge("policy_evidence", "coverage_assessment")
    workflow.add_edge("coverage_assessment", "decision_synthesis")
    workflow.add_edge("decision_synthesis", "validation")
    workflow.add_edge("validation", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("LangGraph state machine compiled successfully.")

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Run the health insurance claim decision graph.")
    parser.add_argument("--pdf-path", default = "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf", help = "Path to the policy PDF to ingest.")
    parser.add_argument("--case-id", default = "PUB-001", help = "Case identifier for the test run.")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    settings = load_settings()
    args = _parse_args()

    graph = build_claim_graph(pdf_path = args.pdf_path, settings = settings)

    # Simulate a fresh run with just the raw data
    test_state = {
        "case_id": args.case_id,
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

    logger.info("--- Starting End-to-End Graph Execution ---")
    final_state = graph.invoke(test_state)

    logger.info("--- End-to-End Execution Complete ---")
    logger.info("Final Decision: %s", final_state.get("final_decision"))
    logger.info("Validation Status: %s", final_state.get("validation_status"))

    print("\n--- Execution Trace ---")
    for step in final_state.get("trace", []):
        print(f"[{step['agent']}] -> {step['action']}")
