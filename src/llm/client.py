"""LLM and Embedding clients wrapping OpenAI-compatible API."""

from __future__ import annotations

from typing import Optional

from openai import OpenAI


class LLMClient:
    """OpenAI-compatible chat completion client."""

    def __init__(self, api_key: str, api_base: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[dict] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 4096,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class EmbeddingClient:
    """OpenAI-compatible embedding client."""

    def __init__(self, api_key: str, api_base: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._model, input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model, input=texts
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
