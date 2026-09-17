from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch

from app.infrastructure.embeddings.local_provider import LocalEmbeddingProvider
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider


class _FakeSentenceTransformer:
    """Stub of sentence_transformers.SentenceTransformer for offline tests."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._counter = 0

    def encode(self, sentences: list[str]) -> np.ndarray:
        vectors = []
        for _sentence in sentences:
            self._counter += 1
            vectors.append([float(self._counter)])
        return np.array(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return 1


class _FakeOpenAIClient:
    """Stub of the OpenAI embeddings client recording each request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self._counter = 0
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> Any:
        self.calls.append((model, input))
        data = []
        for _text in input:
            self._counter += 1
            data.append(SimpleNamespace(embedding=[float(self._counter)]))
        return SimpleNamespace(data=data)


def _patch_openai(monkeypatch: MonkeyPatch) -> _FakeOpenAIClient:
    fake = _FakeOpenAIClient()
    monkeypatch.setattr(
        "app.infrastructure.embeddings.openai_provider.OpenAI",
        lambda api_key: fake,
    )
    return fake


def test_openai_provider_embeds_in_input_order(monkeypatch: MonkeyPatch) -> None:
    fake = _patch_openai(monkeypatch)
    provider = OpenAIEmbeddingProvider(api_key="test-key", batch_size=2048)

    vectors = provider.embed(["a", "b", "c"])

    assert vectors == [[1.0], [2.0], [3.0]]
    assert fake.calls == [("text-embedding-3-small", ["a", "b", "c"])]


def test_openai_provider_auto_batches_large_text_lists(
    monkeypatch: MonkeyPatch,
) -> None:
    fake = _patch_openai(monkeypatch)
    provider = OpenAIEmbeddingProvider(api_key="test-key", batch_size=2)
    texts = ["t0", "t1", "t2", "t3", "t4"]

    vectors = provider.embed(texts)

    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert fake.calls == [
        ("text-embedding-3-small", ["t0", "t1"]),
        ("text-embedding-3-small", ["t2", "t3"]),
        ("text-embedding-3-small", ["t4"]),
    ]


def test_openai_provider_dimension_is_detected_from_response(
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_openai(monkeypatch)
    provider = OpenAIEmbeddingProvider(api_key="test-key", batch_size=2048)

    assert provider.dimension == 1
    assert provider.dimension == 1


def test_openai_provider_uses_configured_model(monkeypatch: MonkeyPatch) -> None:
    fake = _patch_openai(monkeypatch)
    provider = OpenAIEmbeddingProvider(api_key="test-key", model="custom-model")

    provider.embed(["a"])

    assert fake.calls == [("custom-model", ["a"])]


def test_openai_provider_rejects_missing_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenAIEmbeddingProvider(api_key="")


def test_openai_provider_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        OpenAIEmbeddingProvider(api_key="test-key", batch_size=0)


def test_local_provider_embeds_with_mocked_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.embeddings.local_provider.SentenceTransformer",
        _FakeSentenceTransformer,
    )
    provider = LocalEmbeddingProvider(model_name="fake-model", batch_size=2)

    vectors = provider.embed(["a", "b", "c"])

    assert vectors == [[1.0], [2.0], [3.0]]
    assert provider.dimension == 1


def test_local_provider_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        LocalEmbeddingProvider(model_name="")


def test_local_provider_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        LocalEmbeddingProvider(batch_size=0)