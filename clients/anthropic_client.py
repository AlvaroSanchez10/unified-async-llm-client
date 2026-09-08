from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic, APIConnectionError, RateLimitError

from clients.base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):

    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)

        self.client = AsyncAnthropic(api_key=api_key)

    def _format_messages(
        self,
        messages: list[ChatMessage],
    ) -> list[dict[str, str]]:

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
            if message.role != "system"
        ]

    def _get_system_message(
        self,
        messages: list[ChatMessage],
    ) -> str | None:

        system_messages = [
            message.content
            for message in messages
            if message.role == "system"
        ]

        if not system_messages:
            return None

        return "\n".join(system_messages)

    async def generate(
        self,
        messages: list[ChatMessage],
    ) -> ModelResponse:

        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self._get_system_message(messages),
                messages=self._format_messages(messages),
            )

            content = response.content[0].text

            return ModelResponse(
                provider="anthropic",
                model=self.config.model,
                content=content,
            )

        except RateLimitError:
            return ModelResponse(
                provider="anthropic",
                model=self.config.model,
                content="",
                error="Rate limit exceeded.",
            )

        except APIConnectionError:
            return ModelResponse(
                provider="anthropic",
                model=self.config.model,
                content="",
                error="Network error connecting to Anthropic.",
            )

        except Exception as error:
            return ModelResponse(
                provider="anthropic",
                model=self.config.model,
                content="",
                error=str(error),
            )

    async def stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[str]:

        try:
            async with self.client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self._get_system_message(messages),
                messages=self._format_messages(messages),
            ) as stream:

                async for token in stream.text_stream:
                    yield token

        except RateLimitError:
            yield "[ERROR] Rate limit exceeded."

        except APIConnectionError:
            yield "[ERROR] Network error connecting to Anthropic."

        except Exception as error:
            yield f"[ERROR] {error}"
            