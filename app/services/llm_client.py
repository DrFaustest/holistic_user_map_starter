from collections.abc import Iterator
from typing import Protocol

from anthropic import Anthropic
from openai import OpenAI

from app.models.schemas import PromptAssembly
from app.services.settings import LLMSettings


class LLMClient(Protocol):
    def generate_response(self, user_message: str, prompt_assembly: PromptAssembly) -> str:
        ...

    def stream_response(self, user_message: str, prompt_assembly: PromptAssembly) -> Iterator[str]:
        ...


def build_llm_client(settings: LLMSettings) -> LLMClient:
    if settings.provider == "anthropic":
        return AnthropicMessagesClient(settings=settings)
    if settings.provider == "openai":
        return OpenAIChatClient(settings=settings)
    raise ValueError(f"Unsupported LLM_PROVIDER '{settings.provider}'.")


class OpenAIChatClient:
    def __init__(self, settings: LLMSettings):
        resolved_api_key = settings.api_key
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required to use the openai LLM provider.")

        self._model = settings.model
        self._client = OpenAI(api_key=resolved_api_key, base_url=settings.base_url)

    def generate_response(self, user_message: str, prompt_assembly: PromptAssembly) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": prompt_assembly.prompt_instructions,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        )

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ValueError("LLM returned an empty response.")

        return content.strip()

    def stream_response(self, user_message: str, prompt_assembly: PromptAssembly) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": prompt_assembly.prompt_instructions,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicMessagesClient:
    def __init__(self, settings: LLMSettings):
        resolved_api_key = settings.api_key
        if not resolved_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required to use the anthropic LLM provider.")

        self._model = settings.model
        self._client = Anthropic(api_key=resolved_api_key, base_url=settings.base_url)

    def generate_response(self, user_message: str, prompt_assembly: PromptAssembly) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=prompt_assembly.prompt_instructions,
            messages=[{"role": "user", "content": user_message}],
        )
        text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        if not text_blocks:
            raise ValueError("LLM returned an empty response.")
        return "".join(text_blocks).strip()

    def stream_response(self, user_message: str, prompt_assembly: PromptAssembly) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=prompt_assembly.prompt_instructions,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text