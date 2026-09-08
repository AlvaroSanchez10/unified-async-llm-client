import asyncio
import os

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig


load_dotenv()


async def main():

    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        model = "gpt-4o-mini"

    elif provider == "anthropic":
        model = "claude-3-5-haiku-latest"

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    config = ModelConfig(
        provider=provider,
        model=model,
        temperature=0.7,
        max_tokens=200,
    )

    manager = AsyncLLMManager(config)

    messages = [
        ChatMessage(
            role="user",
            content="¿Qué es la entropía?"
        )
    ]

    print("\n=== MODO NORMAL ===\n")

    response = await manager.generate(messages)

    if response.error:
        print(f"Error: {response.error}")
    else:
        print(response.content)

    print("\n=== MODO STREAMING ===\n")

    async for token in manager.stream(messages):
        print(token, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())