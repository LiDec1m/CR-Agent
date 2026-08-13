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
