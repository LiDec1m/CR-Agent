"""LLM and Embedding clients wrapping OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# Matches a fenced code block: ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL
)


class LLMClient:
    """OpenAI-compatible chat completion client."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        max_tokens: int = 49152,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._max_tokens = max_tokens

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
            "max_tokens": self._max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Structured (JSON) chat with repair retries
    # ------------------------------------------------------------------

    @staticmethod
    def strip_code_fence(text: str) -> str:
        """Remove a markdown code fence wrapping the whole response."""
        match = _FENCE_RE.match(text)
        return match.group(1).strip() if match else text.strip()

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_parse_retries: int = 2,
    ) -> Any | None:
        """Chat expecting a JSON object response, with repair retries.

        Strategy:
        1. Ask the model for JSON; strip markdown fences before parsing.
        2. On a parse failure, feed the raw output and the parser error
           back to the model and ask it to fix its own output.
        3. Repeat up to ``max_parse_retries`` repair attempts.
        4. If all attempts fail, log a warning (system prompt tag + raw
           outputs + errors) and return None. Callers fall back to their
           conservative defaults — the pipeline never crashes on a
           malformed LLM response.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_outputs: list[str] = []
        errors: list[str] = []

        for attempt in range(max_parse_retries + 1):
            if attempt > 0:
                # Repair prompt: show what went wrong and ask for a fix.
                repair = (
                    "Your previous response could not be parsed as JSON.\n"
                    f"Parser error: {errors[-1]}\n\n"
                    f"Your previous response:\n{raw_outputs[-1]}\n\n"
                    "Return ONLY a corrected, valid JSON object with no "
                    "markdown fences, no comments, and no extra text. "
                    "Keep the same schema as originally requested."
                )
                messages.append({"role": "assistant", "content": raw_outputs[-1]})
                messages.append({"role": "user", "content": repair})

            raw = self.chat_with_messages(messages)
            raw_outputs.append(raw)

            candidate = self.strip_code_fence(raw)
            # Some models emit a leading BOM or zero-width chars.
            candidate = candidate.lstrip("\ufeff\u200b\u200c\u200d")

            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(str(exc))
                logger.debug(
                    "chat_json parse failed (attempt %d/%d) for %r: %s",
                    attempt + 1, max_parse_retries + 1,
                    system_prompt[:60], exc,
                )

        logger.warning(
            "chat_json gave up after %d attempts for %r; "
            "raw outputs: %r; errors: %r. Falling back to caller defaults.",
            max_parse_retries + 1, system_prompt[:60], raw_outputs, errors,
        )
        return None

    def chat_with_messages(self, messages: list[dict[str, str]]) -> str:
        """Low-level chat over an explicit message list (repair loop)."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""


class EmbeddingClient:
    """OpenAI-compatible embedding client.

    Uses an explicit request timeout and bounded retries so that a
    stalled API connection raises instead of hanging the indexer or
    graph pipeline forever.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=max_retries,
        )
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
