import os
import json
import time
import logging
import requests
from typing import List, Dict, Any


# Configure logging for production-style output
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000/analyze"


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """
        Loads the test cases from a JSON file.
    """
    logger.info(f"Loading test cases from {file_path}...")
    with open(file_path, 'r') as f:
        return json.load(f)


def run_evaluation(cases: List[Dict[str, Any]], output_file: str):
    """
        Runs the test cases through the FastAPI backend and saves the results.
    """
    logger.info(f"Starting evaluation for {len(cases)} cases...")
    
    results = []
    success_count = 0
    needs_review_count = 0
    total_time = 0.0
    
    for case in cases:
        case_id = case.get("case_id", "UNKNOWN")
        logger.info(f"Evaluating case: {case_id}")
        
        start_time = time.time()
        try:
            response = requests.post(BASE_URL, json = case)
            response.raise_for_status()
            data = response.json()
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            decision = data.get("decision")
            val_status = data.get("validation", {}).get("status")
            
            # Track abstentions required by the rubric
            if decision == "NEEDS_REVIEW":
                needs_review_count += 1
            
            # Track citation validation passes
            if val_status == "PASS":
                success_count += 1
            
            results.append({
                "case_id": case_id,
                "expected_task": case.get("task"),
                "actual_decision": decision,
                "confidence": data.get("confidence"),
                "validation_status": val_status,
                "elapsed_time_sec": round(elapsed, 2),
                "citations_count": len(data.get("citations", [])),
                "key_findings": data.get("key_findings", [])
            })
            
            logger.info(f"[{case_id}] Decision: {decision} | Validation: {val_status} | Time: {round(elapsed, 2)}s")
            
        except Exception as e:
            logger.error(f"Failed to evaluate case {case_id}: {e}")
            results.append({"case_id": case_id, "error": str(e)})
            
    # Save results for documentation
    with open(output_file, 'w') as f:
        json.dump(results, f, indent = 4)
        
    logger.info("--- Evaluation Summary ---")
    logger.info(f"Total Cases Processed: {len(cases)}")
    logger.info(f"Validation Pass Rate (Citations Support Decision): {success_count}/{len(cases)}")
    logger.info(f"Abstentions (NEEDS_REVIEW): {needs_review_count} (Requirement: Minimum 2)")
    logger.info(f"Average Time per Case: {round(total_time / max(len(cases), 1), 2)}s")
    logger.info(f"Detailed evaluation saved to {output_file}")


if __name__ == "__main__":
    # Adjust this path if your public_test_cases.json is located elsewhere
    public_cases_path = "data/candidate_data/public_test_cases.json"
    results_path = "tests/test_cases_evaluation_results.json"
    
    if not os.path.exists(public_cases_path):
        logger.error(f"Could not find {public_cases_path}. Ensure you are running from the root directory.")
    else:
        cases = load_test_cases(public_cases_path)
        run_evaluation(cases, results_path)
