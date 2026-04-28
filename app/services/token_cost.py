import re
from dataclasses import dataclass

try:
    import tiktoken
except ImportError:  # pragma: no cover - exercised only when the extra dependency is missing.
    tiktoken = None

from app.services.settings import LLMSettings


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class _FallbackEncoding:
    name = "fallback"

    def encode(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


class ProviderTokenCostEstimator:
    _PRICING_PER_MILLION: dict[str, dict[str, tuple[float, float]]] = {
        "openai": {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
        },
        "anthropic": {
            "claude-3-5-haiku-latest": (0.80, 4.00),
            "claude-3-5-sonnet-latest": (3.00, 15.00),
        },
    }

    def __init__(self, settings: LLMSettings):
        self._settings = settings
        self._encoding = self._build_encoding(settings.provider, settings.model)

    @property
    def tokenizer_name(self) -> str:
        return f"{self._settings.provider}:{self._encoding.name}"

    def estimate_chat_cost(self, system_prompt: str, user_message: str, assistant_response: str) -> CostEstimate:
        input_tokens = self._count_message_tokens(system_prompt, role="system")
        input_tokens += self._count_message_tokens(user_message, role="user")
        output_tokens = self._count_message_tokens(assistant_response, role="assistant")
        total_tokens = input_tokens + output_tokens
        input_rate, output_rate = self._price_for_model(self._settings.provider, self._settings.model)
        estimated_cost_usd = round(
            ((input_tokens * input_rate) + (output_tokens * output_rate)) / 1_000_000,
            8,
        )
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

    def _count_message_tokens(self, text: str, role: str) -> int:
        role_overhead = self._message_overhead(role)
        return len(self._encoding.encode(text)) + role_overhead

    def _build_encoding(self, provider: str, model: str):
        if tiktoken is None:
            return _FallbackEncoding()
        return tiktoken.get_encoding(self._encoding_name(provider, model))

    def _message_overhead(self, role: str) -> int:
        if self._settings.provider == "openai":
            if role == "assistant":
                return 3
            return 4
        if role == "assistant":
            return 5
        return 6

    def _encoding_name(self, provider: str, model: str) -> str:
        normalized_model = model.lower()
        if provider == "openai":
            if "gpt-4o" in normalized_model:
                return "o200k_base"
            return "cl100k_base"
        if provider == "anthropic":
            return "cl100k_base"
        return "cl100k_base"

    def _price_for_model(self, provider: str, model: str) -> tuple[float, float]:
        model_catalog = self._PRICING_PER_MILLION.get(provider, {})
        normalized_model = model.lower()
        for known_model, price in model_catalog.items():
            if known_model in normalized_model:
                return price
        if provider == "openai":
            return 0.15, 0.60
        if provider == "anthropic":
            return 3.00, 15.00
        return 1.00, 5.00