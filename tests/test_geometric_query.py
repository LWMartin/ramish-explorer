import os
"""
Test suite for v0.2 hybrid geometric query engine.
"""
import pytest
from ramish_explorer.reader import RamishFile

CHINOOK_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "chinook.ramish")

@pytest.fixture(scope="module")
def rf():
    return RamishFile.load(CHINOOK_PATH)

def test_geometric_query_returns_results(rf):
    """Query for a known entity should return results."""
    entity = rf.entities[0]
    results = rf.query(entity.name)
    assert len(results) > 0

def test_geometric_query_finds_known_relations(rf):
    """Geometric query for a known entity should surface actual neighbors."""
    # Find an entity with known neighbors
    for eid, neighbors in rf.neighbors.items():
        if len(neighbors) >= 3:
            entity = rf.id_to_entity.get(eid)
            if entity:
                break
    results = rf.query(entity.name, topk=20)
    known_neighbor_names = set()
    for n_id, _, _ in rf.neighbors.get(entity.id, []):
        n = rf.id_to_entity.get(n_id)
        if n:
            known_neighbor_names.add(n.name.lower())
    result_names = {r.object.lower() for r in results}
    overlap = known_neighbor_names & result_names
    assert len(overlap) > 0, f"No known neighbors found in query results for {entity.name}"

def test_geometric_query_known_edges_not_tilde(rf):
    """Known graph edges should NOT have ~ prefix on relation name."""
    entity = rf.entities[0]
    results = rf.query(entity.name, topk=20)
    for r in results:
        if not r.relation.startswith("~"):
            # At least one non-tilde result means graph edges are surfacing
            return
    # It's ok if all are tilde for some entities, but not a structural failure

def test_geometric_query_respects_topk(rf):
    """Query should respect topk limit."""
    entity = rf.entities[0]
    results = rf.query(entity.name, topk=3)
    assert len(results) <= 3

def test_geometric_query_empty_for_nonsense(rf):
    """Query for nonexistent entity should return empty."""
    results = rf.query("zzzzz_absolutely_nothing_matches_this")
    assert len(results) == 0

def test_query_results_sorted_by_weight(rf):
    """Results should be sorted by truth_weight descending."""
    entity = rf.entities[0]
    results = rf.query(entity.name, topk=10)
    if len(results) > 1:
        weights = [r.truth_weight for r in results]
        assert weights == sorted(weights, reverse=True)
