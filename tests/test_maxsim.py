import numpy as np
import pytest

from late_interaction.maxsim import ExactMaxSimIndex, maxsim_score, maxsim_scores


def test_maxsim_matches_hand_computed_token_interactions() -> None:
    query = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    document = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]], dtype=np.float32)

    # Query maxima are 1.0 and 1.0; weak interactions with other tokens are ignored.
    assert maxsim_score(query, document) == pytest.approx(2.0)
    np.testing.assert_allclose(maxsim_scores(query, [document, document]), [2.0, 2.0])


def test_exact_index_ranks_multi_vector_documents_and_breaks_ties() -> None:
    documents = {
        "b-unrelated": np.array([[0.0, 1.0]]),
        "a-related": np.array([[0.2, 0.0], [1.0, 0.0]]),
        "c-same-score": np.array([[1.0, 0.0]]),
    }
    index = ExactMaxSimIndex(documents)

    results = index.search(np.array([[1.0, 0.0]]), k=3)

    assert [result.document_id for result in results] == [
        "a-related",
        "c-same-score",
        "b-unrelated",
    ]
    assert results[0].score == results[1].score == 1.0


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        maxsim_score(np.ones((1, 2)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="non-empty"):
        maxsim_score(np.ones((1, 2)), np.ones((0, 2)))
    with pytest.raises(ValueError, match="same embedding dimension"):
        ExactMaxSimIndex({"a": np.ones((1, 2)), "b": np.ones((1, 3))})
    with pytest.raises(ValueError, match="positive"):
        ExactMaxSimIndex({"a": np.ones((1, 2))}).search(np.ones((1, 2)), k=0)
