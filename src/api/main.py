import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from src.agents.graph import build_claim_graph
from src.config import load_settings, configure_logging
from src.models.schemas import ClaimDecisionContract


# Configure root logging for the application using our centralized config
configure_logging()
logger = logging.getLogger(__name__)


# Global variable to hold our compiled LangGraph application
claim_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        Lifespan context manager for FastAPI. Replaces the deprecated on_event("startup").
        We build and compile the graph once here so we don't have to re-ingest the PDF.
    """
    global claim_graph
    target_pdf = "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    
    logger.info("Starting up API server and compiling LangGraph...")
    try:
        settings = load_settings()
        claim_graph = build_claim_graph(pdf_path = target_pdf, settings = settings)
        logger.info("Graph compiled successfully. API is ready to accept requests.")
    except Exception as e:
        logger.error(f"Failed to initialize graph on startup: {e}")
        raise e
        
    yield  # The FastAPI application runs during this yield
    
    logger.info("Shutting down API server...")


# Initialize FastAPI app with the lifespan manager
app = FastAPI(title = "Aptino Policy-Aware Claim Engine API", version = "1.0.0", lifespan = lifespan)


@app.get("/health")
def health_check():
    """
        Health/readiness check endpoint required by the assignment.
    """
    if claim_graph is None:
        raise HTTPException(status_code = 503, detail = "Graph engine is not initialized.")
    return {"status": "healthy", "engine": "ready"}


@app.post("/analyze", response_model = ClaimDecisionContract)
def analyze_claim(claim_payload: Dict[str, Any]):
    """
        Analyzes one claim case and returns the structured decision.
    """
    case_id = claim_payload.get("case_id", "UNKNOWN_CASE")
    logger.info(f"Received analysis request for case: {case_id}")
    
    if claim_graph is None:
        raise HTTPException(status_code = 503, detail = "Graph engine is not initialized.")
        
    # Prepare the initial state
    initial_state = {
        "case_id": case_id,
        "raw_claim_data": claim_payload
    }
    
    try:
        # Execute the multi-agent graph
        final_state = claim_graph.invoke(initial_state)
        
        # Package the final state into our strict response contract
        response = ClaimDecisionContract(
            case_id = final_state.get("case_id", case_id),
            decision = final_state.get("final_decision", "NEEDS_REVIEW"),
            confidence = final_state.get("confidence", 0.0),
            key_findings = final_state.get("key_findings", []),
            applicable_limits = final_state.get("applicable_limits", []),
            missing_evidence = final_state.get("missing_evidence", []),
            citations = final_state.get("citations", []),
            validation = {
                "status": final_state.get("validation_status", "FAIL"),
                "unsupported_claims": final_state.get("unsupported_claims", [])
            },
            trace = final_state.get("trace", [])
        )
        
        logger.info(f"Successfully processed case {case_id}. Decision: {response.decision}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing claim {case_id}: {str(e)}")
        raise HTTPException(status_code = 500, detail = str(e))


if __name__ == "__main__":
    import uvicorn
    # Run the server locally on port 8000
    uvicorn.run("src.api.main:app", host = "0.0.0.0", port = 8000, reload = True)
