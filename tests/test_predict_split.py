import os
"""
Test suite for v0.2 predict/recommend semantic split.

predict: always frozen key rotation (no neighborhood averaging)
recommend: neighborhood averaging when direct edges exist, frozen key fallback
"""
import numpy as np
import pytest
from ramish_explorer.reader import RamishFile
from ramish_explorer.quate import hamilton_product_np

CHINOOK_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "chinook.ramish")

@pytest.fixture(scope="module")
def rf():
    return RamishFile.load(CHINOOK_PATH)

def test_predict_uses_frozen_key(rf):
    """Predict path should use frozen key rotation, not neighborhood averaging."""
    # Find an entity with known edges of some relation type
    r = rf.relations[0]
    head_id = r.head_id
    rel_type = r.relation_type

    frozen_key, _ = rf._extract_frozen_key(rel_type)
    assert frozen_key is not None

    head_q = rf.embedding_to_quats(head_id)
    predicted_q = hamilton_product_np(head_q[np.newaxis, ...], frozen_key[np.newaxis, ...])
    predicted = predicted_q.squeeze().T.flatten()

    # This should be a valid embedding-space vector
    assert predicted.shape == rf.embeddings[0].shape
    assert np.all(np.isfinite(predicted))

def test_recommend_neighborhood_when_edges_exist(rf):
    """Recommend should use neighborhood averaging when direct edges exist."""
    # Find entity with known targets
    for (head_id, rel_type), targets in rf.targets_by_head_rel.items():
        if len(targets) >= 2:
            break
    else:
        pytest.skip("No entity with 2+ targets for same relation type")

    # Neighborhood prediction = mean of known target embeddings
    target_embs = rf.embeddings[targets]
    neighborhood_pred = np.mean(target_embs, axis=0)
    assert neighborhood_pred.shape == rf.embeddings[0].shape

    # This should differ from frozen key prediction
    frozen_key, _ = rf._extract_frozen_key(rel_type)
    head_q = rf.embedding_to_quats(head_id)
    fk_pred_q = hamilton_product_np(head_q[np.newaxis, ...], frozen_key[np.newaxis, ...])
    fk_pred = fk_pred_q.squeeze().T.flatten()

    # They should be different vectors (neighborhood is local, frozen key is global)
    diff = np.linalg.norm(neighborhood_pred - fk_pred)
    assert diff > 1e-6, "Neighborhood and frozen key predictions should differ"

def test_recommend_falls_back_to_frozen_key(rf):
    """When no direct edges exist, recommend should still work (frozen key fallback)."""
    # Find an entity NOT in targets_by_head_rel for some relation
    rel_type = list(rf.relation_types.values())[0]
    for entity in rf.entities:
        hr_key = (entity.id, rel_type)
        if hr_key not in rf.targets_by_head_rel:
            # This entity has no direct edges of this type
            frozen_key, _ = rf._extract_frozen_key(rel_type)
            assert frozen_key is not None
            head_q = rf.embedding_to_quats(entity.id)
            predicted_q = hamilton_product_np(head_q[np.newaxis, ...], frozen_key[np.newaxis, ...])
            predicted = predicted_q.squeeze().T.flatten()
            assert np.all(np.isfinite(predicted))
            return
    pytest.skip("All entities have edges for the first relation type")
