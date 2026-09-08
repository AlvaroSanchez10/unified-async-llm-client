import os
from collections.abc import AsyncIterator

from clients.anthropic_client import AnthropicClient
from clients.base import BaseLLMClient
from clients.openai_client import OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class AsyncLLMManager:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = self._create_client()

    def _create_client(self) -> BaseLLMClient:

        if self.config.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not configured."
                )

            return OpenAIClient(
                config=self.config,
                api_key=api_key,
            )

        if self.config.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")

            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not configured."
                )

            return AnthropicClient(
                config=self.config,
                api_key=api_key,
            )

        raise ValueError(
            f"Unsupported provider: {self.config.provider}"
        )

    async def generate(
        self,
        messages: list[ChatMessage],
    ) -> ModelResponse:

        return await self.client.generate(messages)

    async def stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[str]:

        async for token in self.client.stream(messages):
            yield token