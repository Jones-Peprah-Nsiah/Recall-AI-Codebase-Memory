import threading

import numpy as np
from fastembed import TextEmbedding

from .config import settings

_model: TextEmbedding | None = None
_lock = threading.Lock()


def model() -> TextEmbedding:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = TextEmbedding(settings.embed_model)
    return _model


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    return list(model().passage_embed(texts, batch_size=settings.embed_batch))


def embed_query(text: str) -> np.ndarray:
    # query_embed adds the model's retrieval instruction prefix (bge uses one).
    return next(iter(model().query_embed(text))).astype(np.float32)
