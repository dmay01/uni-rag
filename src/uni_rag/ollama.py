"""Only HTTP client in the runtime: loopback Ollama, no proxies or redirects."""

import json
from collections.abc import Iterator

import httpx

from uni_rag.config import Config


class Ollama:
    def __init__(self, config: Config, transport: httpx.BaseTransport | None = None):
        self.config = config
        self.model_info: dict[str, dict] = {}
        self.model_digests: dict[str, str] = {}
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

    def thinking_mode(self, model: str, requested: bool) -> bool:
        if model not in self.model_info:
            response = self.client.post("/api/show", json={"model": model})
            response.raise_for_status()
            self.model_info[model] = response.json()
        info = self.model_info[model].get("model_info", {})
        return requested or str(info.get("general.finetune", "")).casefold() == "thinking"

    def digest(self, model: str) -> str | None:
        if model not in self.model_digests:
            response = self.client.get("/api/tags")
            response.raise_for_status()
            self.model_digests.update(
                {item["name"]: item["digest"] for item in response.json()["models"]}
            )
        return self.model_digests.get(model)

    def chat(self, messages: list[dict], model: str, think: bool = False) -> Iterator[dict]:
        effective = self.thinking_mode(model, think)
        if not effective:
            messages = [dict(message) for message in messages]
            messages[0]["content"] += "\n/no_think"
        with self.client.stream(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "think": effective,
                "logprobs": True,
                "top_logprobs": 0,
                "options": {
                    "num_ctx": self.config.context_window,
                    "num_predict": self.config.num_predict,
                },
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
