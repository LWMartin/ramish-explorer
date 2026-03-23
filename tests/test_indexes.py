import os
"""
Test suite for v0.2 relation indexes.
Validates edges_by_rel_type, out_edges, in_edges, compound lookups.
"""
import pytest
from ramish_explorer.reader import RamishFile

CHINOOK_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "chinook.ramish")

@pytest.fixture(scope="module")
def rf():
    return RamishFile.load(CHINOOK_PATH)

def test_edges_by_rel_type_completeness(rf):
    """Every relation should appear in exactly one rel_type bucket."""
    total = sum(len(v) for v in rf.edges_by_rel_type.values())
    assert total == len(rf.relations)

def test_edges_by_rel_type_no_duplicates(rf):
    """No relation index should appear twice."""
    all_indices = []
    for indices in rf.edges_by_rel_type.values():
        all_indices.extend(indices)
    assert len(all_indices) == len(set(all_indices))

def test_out_edges_consistent(rf):
    """out_edges[head][rel_type] should only contain relations with that head+type."""
    for head_id, rel_dict in list(rf.out_edges.items())[:50]:
        for rel_type, indices in rel_dict.items():
            for idx in indices:
                r = rf.relations[idx]
                assert r.head_id == head_id
                assert r.relation_type == rel_type

def test_in_edges_consistent(rf):
    """in_edges[tail][rel_type] should only contain relations with that tail+type."""
    for tail_id, rel_dict in list(rf.in_edges.items())[:50]:
        for rel_type, indices in rel_dict.items():
            for idx in indices:
                r = rf.relations[idx]
                assert r.tail_id == tail_id
                assert r.relation_type == rel_type

def test_targets_by_head_rel_matches_scan(rf):
    """Compound index should match a brute-force scan."""
    r = rf.relations[0]
    key = (r.head_id, r.relation_type)
    indexed = set(rf.targets_by_head_rel.get(key, []))
    scanned = {rel.tail_id for rel in rf.relations
               if rel.head_id == r.head_id and rel.relation_type == r.relation_type}
    assert indexed == scanned

def test_heads_by_tail_rel_matches_scan(rf):
    """Reverse compound index should match brute-force scan."""
    r = rf.relations[0]
    key = (r.tail_id, r.relation_type)
    indexed = set(rf.heads_by_tail_rel.get(key, []))
    scanned = {rel.head_id for rel in rf.relations
               if rel.tail_id == r.tail_id and rel.relation_type == r.relation_type}
    assert indexed == scanned

def test_embedding_norms_precomputed(rf):
    """Embedding norms should be precomputed and match manual calculation."""
    import numpy as np
    assert rf._embedding_norms is not None
    assert len(rf._embedding_norms) == len(rf.entities)
    # Spot check first 10
    manual = np.linalg.norm(rf.embeddings[:10], axis=1)
    np.testing.assert_allclose(rf._embedding_norms[:10], manual)
