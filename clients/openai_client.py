from collections.abc import AsyncIterator

from openai import AsyncOpenAI, APIConnectionError, RateLimitError

from clients.base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):

    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)

        self.client = AsyncOpenAI(api_key=api_key)

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
        ]

    async def generate(
        self,
        messages: list[ChatMessage],
    ) -> ModelResponse:

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=self._format_messages(messages),
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
            )

            content = response.choices[0].message.content or ""

            return ModelResponse(
                provider="openai",
                model=self.config.model,
                content=content,
            )

        except RateLimitError:
            return ModelResponse(
                provider="openai",
                model=self.config.model,
                content="",
                error="Rate limit exceeded.",
            )

        except APIConnectionError:
            return ModelResponse(
                provider="openai",
                model=self.config.model,
                content="",
                error="Network error connecting to OpenAI.",
            )

        except Exception as error:
            return ModelResponse(
                provider="openai",
                model=self.config.model,
                content="",
                error=str(error),
            )

    async def stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[str]:

        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=self._format_messages(messages),
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                token = chunk.choices[0].delta.content

                if token:
                    yield token

        except RateLimitError:
            yield "[ERROR] Rate limit exceeded."

        except APIConnectionError:
            yield "[ERROR] Network error connecting to OpenAI."

        except Exception as error:
            yield f"[ERROR] {error}"