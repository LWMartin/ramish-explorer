import os
"""
Test suite for v0.2 embedding_to_quats canonical helper.
"""
import numpy as np
import pytest
from ramish_explorer.reader import RamishFile

CHINOOK_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "chinook.ramish")

@pytest.fixture(scope="module")
def rf():
    return RamishFile.load(CHINOOK_PATH)

def test_single_entity_shape(rf):
    """Single entity should return (dim, 4)."""
    q = rf.embedding_to_quats(0)
    dim = rf.embeddings.shape[1] // 4
    assert q.shape == (dim, 4)

def test_batch_entity_shape(rf):
    """Batch should return (n, dim, 4)."""
    ids = [0, 1, 2]
    q = rf.embedding_to_quats(ids)
    dim = rf.embeddings.shape[1] // 4
    assert q.shape == (3, dim, 4)

def test_single_matches_batch(rf):
    """Single entity result should match corresponding batch row."""
    single = rf.embedding_to_quats(0)
    batch = rf.embedding_to_quats([0, 1, 2])
    np.testing.assert_allclose(single, batch[0])

def test_roundtrip(rf):
    """embedding_to_quats then flatten back should give original embedding."""
    eid = 5
    q = rf.embedding_to_quats(eid)  # (dim, 4)
    # Reverse: T then flatten -> (4, dim) -> flatten
    recovered = q.T.flatten()
    np.testing.assert_allclose(recovered, rf.embeddings[eid], atol=1e-7)

def test_numpy_array_input(rf):
    """Should accept numpy array of ids."""
    ids = np.array([0, 1, 2])
    q = rf.embedding_to_quats(ids)
    dim = rf.embeddings.shape[1] // 4
    assert q.shape == (3, dim, 4)
