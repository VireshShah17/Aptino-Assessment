import logging
from typing import Any, Dict, Optional, Type, TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.config import Settings


logger = logging.getLogger(__name__)
T = TypeVar("T", bound = BaseModel)


class BaseLLMAgent:
    """
        Common scaffolding for agents that call a structured-output LLM.
    """

    def __init__(self, settings: Settings, output_schema: Type[T], system_prompt: str, human_prompt: str, agent_name: str):
        logger.info("Initializing %s...", agent_name)
        self.agent_name = agent_name
        self.output_schema = output_schema

        base_llm = ChatGoogleGenerativeAI(
            model = settings.model_name,
            temperature = settings.temperature,
            google_api_key = settings.google_api_key,
            max_retries = settings.max_retries,
            timeout = settings.request_timeout,
        )
        self.llm = base_llm.with_structured_output(output_schema)

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )
        self.chain = self.prompt | self.llm
        logger.info("%s initialized successfully.", agent_name)


    def _safe_invoke(self, inputs: Dict[str, Any]) -> Optional[T]:
        """
            Invoke the chain, returning None (and logging) on any failure.
            This is the single error-handling chokepoint for every agent's
            LLM call - covers network errors, timeouts, rate limits, and
            structured-output validation failures. Callers decide what
            fallback state update to return when this yields None (typically
            a NEEDS_REVIEW-style result rather than letting the exception
            propagate and crash the whole graph run).
        """
        try:
            return self.chain.invoke(inputs)
        except Exception:
            logger.exception("%s: LLM invocation failed.", self.agent_name)
            return None

    
    def trace_entry(self, action: str) -> Dict[str, str]:
        return {"agent": self.agent_name, "action": action}
