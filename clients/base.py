from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
    ) -> ModelResponse:
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[str]:
        yield ""
        