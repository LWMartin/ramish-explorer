import os
"""
Test suite for v0.2 sign-aligned frozen key extraction.
"""
import numpy as np
import pytest
from ramish_explorer.reader import RamishFile
from ramish_explorer.quate import hamilton_product_np, quaternion_conjugate_np

CHINOOK_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "chinook.ramish")

@pytest.fixture(scope="module")
def rf():
    return RamishFile.load(CHINOOK_PATH)

def test_frozen_key_returns_tuple(rf):
    """_extract_frozen_key should return (ndarray, float)."""
    rel_type = list(rf.relation_types.values())[0]
    key, stability = rf._extract_frozen_key(rel_type)
    assert key is not None
    assert isinstance(stability, float)

def test_frozen_key_shape(rf):
    """Frozen key should be (dim, 4)."""
    rel_type = list(rf.relation_types.values())[0]
    key, _ = rf._extract_frozen_key(rel_type)
    dim = rf.embeddings.shape[1] // 4
    assert key.shape == (dim, 4)

def test_frozen_key_stability_bounded(rf):
    """Stability should be in [0, 1]."""
    for rel_type in rf.relation_types.values():
        _, stability = rf._extract_frozen_key(rel_type)
        assert 0.0 <= stability <= 1.0

def test_frozen_key_cached(rf):
    """Second call should return same object (cached)."""
    rel_type = list(rf.relation_types.values())[0]
    key1, _ = rf._extract_frozen_key(rel_type)
    key2, _ = rf._extract_frozen_key(rel_type)
    assert key1 is key2  # same object, not just equal

def test_sign_aligned_key_norm_ge_naive(rf):
    """Sign-aligned frozen key should have >= norm of naive average."""
    for rel_type in list(rf.relation_types.values())[:3]:
        aligned_key, _ = rf._extract_frozen_key(rel_type)
        # Compute naive (no sign alignment) for comparison
        indices = rf.edges_by_rel_type.get(rel_type, [])
        if len(indices) < 2:
            continue
        head_ids = [rf.relations[i].head_id for i in indices]
        tail_ids = [rf.relations[i].tail_id for i in indices]
        h_q = rf.embedding_to_quats(head_ids)
        t_q = rf.embedding_to_quats(tail_ids)
        naive_keys = hamilton_product_np(quaternion_conjugate_np(h_q), t_q)
        naive_mean = np.mean(naive_keys, axis=0)
        aligned_norm = np.linalg.norm(aligned_key)
        naive_norm = np.linalg.norm(naive_mean)
        assert aligned_norm >= naive_norm - 1e-6, \
            f"Aligned norm {aligned_norm:.4f} < naive norm {naive_norm:.4f}"

def test_nonexistent_rel_type(rf):
    """Non-existent relation type should return (None, 0.0)."""
    key, stability = rf._extract_frozen_key(99999)
    assert key is None
    assert stability == 0.0
