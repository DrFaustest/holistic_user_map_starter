import os

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    provider: str = Field(default="openai")
    api_key: str | None = Field(default=None)
    model: str = Field(default="gpt-4o-mini")
    base_url: str | None = Field(default=None)


def _provider_value(provider: str, openai_value: str | None, anthropic_value: str | None) -> str | None:
    if provider == "anthropic":
        return anthropic_value
    return openai_value


def load_llm_settings() -> LLMSettings:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    return LLMSettings(
        provider=provider,
        api_key=_provider_value(
            provider,
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
        ),
        model=_provider_value(
            provider,
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        ) or "gpt-4o-mini",
        base_url=_provider_value(
            provider,
            os.getenv("OPENAI_BASE_URL"),
            os.getenv("ANTHROPIC_BASE_URL"),
        ),
    )