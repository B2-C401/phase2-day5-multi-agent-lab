"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# gpt-4o-mini pricing as of 2024: $0.15/1M input, $0.60/1M output
_INPUT_COST_PER_TOKEN = 0.000_000_150
_OUTPUT_COST_PER_TOKEN = 0.000_000_600


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client wrapping OpenAI."""

    def __init__(self) -> None:
        settings = get_settings()
        from openai import OpenAI  # lazy import so tests don't need the package

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._timeout = settings.timeout_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry on transient errors."""

        logger.debug("LLM call model=%s", self._model)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self._timeout,
        )
        content = resp.choices[0].message.content or ""
        in_tok = resp.usage.prompt_tokens if resp.usage else None
        out_tok = resp.usage.completion_tokens if resp.usage else None
        cost: float | None = None
        if in_tok is not None and out_tok is not None:
            cost = in_tok * _INPUT_COST_PER_TOKEN + out_tok * _OUTPUT_COST_PER_TOKEN
        logger.debug(
            "LLM done in_tokens=%s out_tokens=%s cost_usd=%.6f", in_tok, out_tok, cost or 0.0
        )
        return LLMResponse(content=content, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
