from typing import Any, Dict, List
from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str = Field(description = "The ID of the policy evidence chunk that supports the claim.")
    claim: str = Field(description = "The specific claim being supported by this citation.")


class AnalysisOutput(BaseModel):
    extracted_facts: Dict[str, Any] = Field(description = "Key facts extracted from the raw claim data (e.g., diagnosis, dates, amounts).")
    investigation_plan: List[str] = Field(description = "Checklist of policy dimensions to investigate (e.g., waiting periods, specific limits, exclusions).")


class CoverageFindings(BaseModel):
    """
        Defaults are important here: they let a code-level fallback (e.g. "no
        evidence was retrieved") construct a fully-populated instance instead
        of a partial dict, so downstream agents can always rely on every field
        being present regardless of which code path produced it.
    """
    is_covered: bool = Field(default = False, description = "True if the base treatment is covered, False if strictly excluded or in a waiting period.")
    applicable_limits: List[str] = Field(default_factory = list, description = "Any limits, caps, or deductions found in the evidence (e.g., Room rent cap at 1%).")
    waiting_periods_applied: List[str] = Field(default_factory = list, description = "Any waiting periods that apply and invalidate the claim.")
    reasoning: str = Field(default = "", description = "Brief explanation of the coverage decision based strictly on evidence.")
    needs_review: bool = Field(default = True, description = "True if the evidence is insufficient to make a safe conclusion.")


class DecisionOutput(BaseModel):
    decision: str = Field(description = (
            "Must be exactly one of: ADMISSIBLE, ADMISSIBLE_WITH_LIMITS, "
            "PARTIALLY_ADMISSIBLE, NOT_ADMISSIBLE, NEEDS_REVIEW"
        ))
    confidence: float = Field(description = "Confidence score between 0.0 and 1.0")
    key_findings: List[str] = Field(description = "Summary of major reasons for the decision")
    applicable_limits: List[str] = Field(description = "Any limits or deductions that affect the payable amount")
    missing_evidence: List[str] = Field(description = "Any required evidence that is missing, prompting a NEEDS_REVIEW status")
    citations: List[Citation] = Field(description = "Specific citations to policy evidence supporting the material claims")


class ValidationResult(BaseModel):
    status: str = Field(description = "Either 'PASS' or 'FAIL'.")
    unsupported_claims: List[str] = Field(default_factory = list, description = "Claims that are not supported by the retrieved evidence.")


class ValidationResponse(BaseModel):
    status: str
    unsupported_claims: List[str]


class ClaimDecisionContract(BaseModel):
    case_id: str
    decision: str = Field(description = "e.g., ADMISSIBLE, ADMISSIBLE_WITH_LIMITS, NEEDS_REVIEW")
    confidence: float
    key_findings: List[str]
    applicable_limits: List[str]
    missing_evidence: List[str]
    citations: List[Citation]
    validation: ValidationResponse
    trace: List[Dict[str, Any]]


VALID_DECISIONS = {
    "ADMISSIBLE",
    "ADMISSIBLE_WITH_LIMITS",
    "PARTIALLY_ADMISSIBLE",
    "NOT_ADMISSIBLE",
    "NEEDS_REVIEW",
}