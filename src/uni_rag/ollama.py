"""Only HTTP client in the runtime: loopback Ollama, no proxies or redirects."""

import json
from collections.abc import Iterator

import httpx

from uni_rag.config import Config


class Ollama:
    def __init__(self, config: Config, transport: httpx.BaseTransport | None = None):
        self.config = config
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )

    def __enter__(self) -> "Ollama":
        return self

    def __exit__(self, *args: object) -> None:
        self.client.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.post(
            "/api/embed",
            json={"model": self.config.embedding_model, "input": texts, "truncate": False},
        )
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        if len(vectors) != len(texts) or not vectors or not vectors[0]:
            raise ValueError("Ollama returned an invalid embedding batch")
        return vectors

    def chat(self, messages: list[dict], model: str, think: bool = False) -> Iterator[dict]:
        with self.client.stream(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "think": think,
                "options": {"num_ctx": self.config.context_window},
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    event = json.loads(line)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    yield event

    def unload(self, model: str) -> None:
        response = self.client.post("/api/generate", json={"model": model, "keep_alive": 0})
        response.raise_for_status()
