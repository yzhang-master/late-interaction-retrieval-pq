"""ColBERT-style late interaction and exact multi-vector retrieval."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def _embedding_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [tokens, dimensions] matrix")
    return matrix


def maxsim_score(query: np.ndarray, document: np.ndarray) -> float:
    """Sum each query token's maximum dot product with any document token."""
    query = _embedding_matrix(query, "query")
    document = _embedding_matrix(document, "document")
    if query.shape[1] != document.shape[1]:
        raise ValueError("query and document embedding dimensions must match")
    return float((query @ document.T).max(axis=1).sum())


def maxsim_scores(query: np.ndarray, documents: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([maxsim_score(query, document) for document in documents], dtype=np.float32)


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    score: float


class ExactMaxSimIndex:
    """Brute-force exact late interaction over variable-length documents."""

    def __init__(self, documents: Mapping[str, np.ndarray]) -> None:
        if not documents:
            raise ValueError("at least one document is required")
        self.document_ids = tuple(documents)
        self.documents = tuple(
            _embedding_matrix(documents[document_id], document_id)
            for document_id in self.document_ids
        )
        dimensions = {document.shape[1] for document in self.documents}
        if len(dimensions) != 1:
            raise ValueError("all documents must have the same embedding dimension")

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if k < 1:
            raise ValueError("k must be positive")
        scores = maxsim_scores(query, self.documents)
        order = sorted(
            range(len(scores)), key=lambda index: (-float(scores[index]), self.document_ids[index])
        )
        return [
            SearchResult(self.document_ids[index], float(scores[index]))
            for index in order[: min(k, len(scores))]
        ]
