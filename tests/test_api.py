import json
import logging
import requests


# Configure logging for production-style output
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    """
        Tests the /health endpoint.
    """
    logger.info("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        logger.info(f"Health Check Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check failed: {e}")


def test_analyze():
    """
        Tests the /analyze endpoint with the PUB-001 claim case.
    """
    logger.info("Testing /analyze endpoint with PUB-001...")
    
    # Simulating the PUB-001 public test case payload
    payload = {
        "case_id": "PUB-001",
        "policy_id": "USGIC-CSC-2017-2018",
        "policy_start_date": "2025-01-01",
        "claim_date": "2026-03-14",
        "sum_insured_inr": 500000,
        "treatment": {
            "type": "inpatient",
            "diagnosis": "Acute appendicitis",
            "procedure": "Appendectomy",
            "pre_existing": False
        },
        "expenses_inr": {
            "room": 30000,
            "doctor_fees": 30000,
            "medicines_diagnostics": 90000
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", json = payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"--- API Response for {result.get('case_id')} ---")
        logger.info(f"Decision: {result.get('decision')}")
        logger.info(f"Confidence: {result.get('confidence')}")
        logger.info(f"Validation Status: {result.get('validation', {}).get('status')}")
        logger.info(f"Number of Citations: {len(result.get('citations', []))}")
        
        print("\nFull JSON Response:")
        print(json.dumps(result, indent = 2))
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Analyze request failed: {e}")


if __name__ == "__main__":
    test_health()
    print("-" * 40)
    test_analyze()
