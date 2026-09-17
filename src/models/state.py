import operator
from typing import Any, Dict, List, TypedDict, Annotated


class ClaimGraphState(TypedDict, total = False):
    case_id: str
    raw_claim_data: Dict[str, Any]

    extracted_facts: Dict[str, Any]
    investigation_plan: List[str]

    retrieved_evidence: List[Any]  # List[langchain_core.documents.Document]

    coverage_findings: Dict[str, Any]

    final_decision: str
    confidence: float
    key_findings: List[str]
    applicable_limits: List[str]
    missing_evidence: List[str]
    citations: List[Dict[str, Any]]

    validation_status: str
    unsupported_claims: List[str]

    trace: Annotated[List[Dict[str, str]], operator.add]
