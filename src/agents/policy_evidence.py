"""
    Policy Evidence Agent: executes the investigation plan against the hybrid
    retriever and gathers unique supporting evidence chunks.
"""
import logging

from src.retrieval.hybrid_search import HybridRetriever


logger = logging.getLogger(__name__)


class PolicyEvidenceAgent:

    def __init__(self, retriever: HybridRetriever):
        """
            Initializes the Evidence Agent with the pre-loaded Hybrid Retriever.
        """
        logger.info("Initializing Policy Evidence Agent...")
        self.retriever = retriever
        logger.info("Policy Evidence Agent initialized successfully.")


    def invoke(self, state: dict) -> dict:
        """
            Executes the investigation plan against the Hybrid Retriever and updates the state.
        """
        case_id = state.get("case_id", "UNKNOWN")
        logger.info("Gathering policy evidence for case: %s", case_id)

        investigation_plan = state.get("investigation_plan", [])
        if not investigation_plan:
            logger.warning("No investigation plan found in state. Skipping retrieval.")
            return {
                "retrieved_evidence": [],
                "trace": [{"agent": "Policy Evidence", "action": "No plan to execute"}],
            }

        all_retrieved_docs = {}

        # Execute hybrid search for each item in the investigation plan
        for query in investigation_plan:
            logger.info("Querying retriever for: '%s'", query)
            try:
                # We fetch top 2 per query to keep the context window tight and highly relevant
                docs = self.retriever.get_fused_results(query = query, k = 2)
            except Exception:
                # A single failed query shouldn't take down the whole
                # evidence-gathering step - log it and keep going with
                # whatever other queries succeed.
                logger.exception("Retriever failed for query '%s'; skipping it.", query)
                continue

            for doc in docs:
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id not in all_retrieved_docs:
                    all_retrieved_docs[chunk_id] = doc

        unique_evidence = list(all_retrieved_docs.values())
        logger.info("Successfully gathered %d unique policy chunks.", len(unique_evidence))

        return {
            "retrieved_evidence": unique_evidence,
            "trace": [
                {
                    "agent": "Policy Evidence",
                    "action": f"Retrieved {len(unique_evidence)} unique clauses based on investigation plan",
                }
            ],
        }


if __name__ == "__main__":
    from src.config import configure_logging
    from src.ingestion.chunking import ingest_policy

    configure_logging()

    # 1. Setup the Retriever (Reusing Sprint 1 logic)
    target_pdf = "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    chunks = ingest_policy(target_pdf)
    hybrid_retriever = HybridRetriever(chunks = chunks)

    # 2. Simulate the state passed from the Case Analysis Agent
    test_state = {
        "case_id": "PUB-001",
        "investigation_plan": [
            "Verify policy is active as of the claim date",
            "Review room rent capping and limits",
            "Check for any specific waiting periods for Appendectomy",
        ],
    }

    # 3. Run the Evidence Agent
    agent = PolicyEvidenceAgent(retriever = hybrid_retriever)
    output_state = agent.invoke(state = test_state)

    logger.info("--- Retrieved Evidence Summary ---")
    for idx, doc in enumerate(output_state["retrieved_evidence"]):
        logger.info("Evidence %d Metadata: %s", idx + 1, doc.metadata)
        logger.info("Evidence %d Snippet: %s...", idx + 1, doc.page_content[:100])

    logger.info("Trace: %s", output_state["trace"])
