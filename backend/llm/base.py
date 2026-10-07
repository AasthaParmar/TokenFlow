from typing import Protocol


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class GenerationClient(EmbeddingClient, Protocol):
    def generate(
        self,
        message: str,
        model: str | None = None,
        context: str | None = None,
        temperature: float = 0.7,
    ):
        ...

    def judge(self, prompt: str) -> str:
        ...