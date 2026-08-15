import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.client import LLMClient, EmbeddingClient


def test_llm_client_chat_returns_string():
    client = LLMClient(api_key="fake", api_base="http://localhost", model="gpt-4o-mini")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hello world"))]
    with patch.object(client._client.chat.completions, "create", return_value=mock_response):
        result = client.chat("system", "user")
        assert result == "hello world"


def test_llm_client_chat_with_json_format():
    client = LLMClient(api_key="fake", api_base="http://localhost", model="gpt-4o-mini")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"key": "value"}'))]
    with patch.object(client._client.chat.completions, "create", return_value=mock_response) as mock_create:
        result = client.chat("system", "user", response_format={"type": "json_object"})
        assert result == '{"key": "value"}'
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_embedding_client_embed_returns_list():
    client = EmbeddingClient(api_key="fake", api_base="http://localhost", model="text-embedding-3-small")
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    with patch.object(client._client.embeddings, "create", return_value=mock_response):
        result = client.embed("hello")
        assert result == [0.1, 0.2, 0.3]


def test_embedding_client_embed_batch():
    client = EmbeddingClient(api_key="fake", api_base="http://localhost", model="text-embedding-3-small")
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2], index=0),
        MagicMock(embedding=[0.3, 0.4], index=1),
    ]
    with patch.object(client._client.embeddings, "create", return_value=mock_response):
        result = client.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]


def _make_client_with_responses(responses):
    """Build an LLMClient whose chat_with_messages returns canned outputs."""
    client = LLMClient(api_key="fake", api_base="http://localhost", model="gpt-4o-mini")
    outputs = list(responses)

    def fake_create(**kwargs):
        content = outputs.pop(0)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=content))]
        return resp

    patcher = patch.object(
        client._client.chat.completions, "create", side_effect=fake_create
    )
    patcher.start()
    return client, patcher


def test_chat_json_parses_plain_response():
    client, patcher = _make_client_with_responses(['{"plan": ["sql_rule"]}'])
    try:
        result = client.chat_json("sys", "user")
        assert result == {"plan": ["sql_rule"]}
    finally:
        patcher.stop()


def test_chat_json_strips_code_fence():
    client, patcher = _make_client_with_responses(
        ['```json\n{"plan": ["a"]}\n```']
    )
    try:
        result = client.chat_json("sys", "user")
        assert result == {"plan": ["a"]}
    finally:
        patcher.stop()


def test_chat_json_repairs_then_succeeds():
    # First two outputs invalid, third valid — must not call default fallback.
    client, patcher = _make_client_with_responses(
        ['oops not json', '{"plan": ', '{"plan": ["fixed"]}']
    )
    try:
        result = client.chat_json("sys", "user", max_parse_retries=3)
        assert result == {"plan": ["fixed"]}
    finally:
        patcher.stop()


def test_chat_json_returns_none_and_logs_after_retries_exhausted():
    client, patcher = _make_client_with_responses(
        ["bad1", "bad2", "bad3", "bad4"]
    )
    try:
        with patch("src.llm.client.logger") as mock_logger:
            result = client.chat_json("sys", "user", max_parse_retries=3)
        assert result is None
        # 1 initial + 3 repairs = 4 calls, then warning logged
        assert mock_logger.warning.called
    finally:
        patcher.stop()
