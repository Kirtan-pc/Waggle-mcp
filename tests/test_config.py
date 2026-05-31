import pytest

from waggle.config import AppConfig
from waggle.errors import ValidationFailure


def test_embedding_backend_defaults_to_pytorch(monkeypatch):
    monkeypatch.delenv("WAGGLE_EMBEDDING_BACKEND", raising=False)

    config = AppConfig.from_env()

    assert config.embedding_backend == "pytorch"


def test_embedding_backend_accepts_onnx(monkeypatch):
    monkeypatch.setenv("WAGGLE_EMBEDDING_BACKEND", "onnx")

    config = AppConfig.from_env()

    assert config.embedding_backend == "onnx"


def test_embedding_backend_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("WAGGLE_EMBEDDING_BACKEND", "banana")

    with pytest.raises(ValidationFailure):
        AppConfig.from_env()