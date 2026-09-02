import sys
from types import SimpleNamespace

from app.config import Settings
from app.knowledge import indexer as indexer_module


def test_settings_exposes_local_retrieval_model_options():
    settings = Settings(
        _env_file=None,
        LOCAL_EMBEDDING_MODEL_PATH="D:/models/bge-m3",
        LOCAL_EMBEDDING_DEVICE="cuda",
        LOCAL_EMBEDDING_BATCH_SIZE=4,
        LOCAL_EMBEDDING_DIMENSION=1024,
        RERANKER_MODEL_PATH="D:/models/bge-reranker-v2-m3",
        RERANKER_DEVICE="cuda",
        RERANKER_BATCH_SIZE=4,
        RERANKER_CANDIDATE_K=20,
    )

    assert settings.LOCAL_EMBEDDING_MODEL_PATH == "D:/models/bge-m3"
    assert settings.LOCAL_EMBEDDING_DEVICE == "cuda"
    assert settings.LOCAL_EMBEDDING_BATCH_SIZE == 4
    assert settings.LOCAL_EMBEDDING_DIMENSION == 1024
    assert settings.RERANKER_MODEL_PATH == "D:/models/bge-reranker-v2-m3"
    assert settings.RERANKER_DEVICE == "cuda"
    assert settings.RERANKER_BATCH_SIZE == 4
    assert settings.RERANKER_CANDIDATE_K == 20


def test_local_embedding_uses_configured_path_device_and_batch_size(monkeypatch):
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, device=None):
            captured["model_name"] = model_name
            captured["device"] = device

        def encode(self, texts, **kwargs):
            captured["texts"] = texts
            captured["encode_kwargs"] = kwargs
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setattr(
        indexer_module,
        "get_settings",
        lambda: SimpleNamespace(
            EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5",
            LOCAL_EMBEDDING_MODEL_PATH="D:/models/bge-m3",
            LOCAL_EMBEDDING_DEVICE="cuda",
            LOCAL_EMBEDDING_BATCH_SIZE=4,
        ),
    )

    embedder = indexer_module.LocalSentenceTransformerEmbeddingFunction()
    result = embedder(["证据一", "证据二"])

    assert captured["model_name"] == "D:/models/bge-m3"
    assert captured["device"] == "cuda"
    assert captured["encode_kwargs"]["batch_size"] == 4
    assert result == [[0.1, 0.2], [0.1, 0.2]]
