import os
"""
Test suite for ramish-explorer
===============================
Validates core functionality of the ramish-explorer package:
- File loading and structure
- Entity and relation integrity
- Quaternion math (Hamilton product, conjugate)
- Trust weight bounds
- Audit pipeline
- Claim validation
- Query and navigation

Fixture: chinook.ramish (Chinook music database, 4652 entities, 22289 relations)
"""

import numpy as np
import pytest

from ramish_explorer.models import Entity, Relation
from ramish_explorer.quate import hamilton_product_np, quaternion_conjugate_np
from ramish_explorer.reader import RamishFile


# ── Fixtures ─────────────────────────────────────────────────────

CHINOOK_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "chinook.ramish")


@pytest.fixture(scope="module")
def rf():
    """Load chinook.ramish once for all tests."""
    return RamishFile.load(CHINOOK_PATH)


# ── 1. File Loading & Structure ──────────────────────────────────


def test_load_returns_ramish_file(rf):
    assert isinstance(rf, RamishFile)


def test_entity_count(rf):
    assert len(rf.entities) == 4652


def test_relation_count(rf):
    assert len(rf.relations) == 22289


def test_embeddings_shape(rf):
    assert rf.embeddings is not None
    assert rf.embeddings.shape[0] == len(rf.entities)
    # 16 quaternion dimensions * 4 components = 64
    assert rf.embeddings.shape[1] == 64


def test_truth_weights_shape(rf):
    assert rf.truth_weights is not None
    assert rf.truth_weights.shape[0] == len(rf.relations)


def test_entity_types_present(rf):
    types = {e.entity_type for e in rf.entities}
    assert len(types) > 0, "Should have at least one entity type"


def test_relation_types_present(rf):
    assert len(rf.relation_type_names) > 0, "Should have named relation types"


# ── 2. Entity & Relation Integrity ──────────────────────────────


def test_entities_are_entity_objects(rf):
    for e in rf.entities[:10]:
        assert isinstance(e, Entity)
        assert isinstance(e.id, int)
        assert isinstance(e.name, str)
        assert len(e.name) > 0


def test_relations_are_relation_objects(rf):
    for r in rf.relations[:10]:
        assert isinstance(r, Relation)
        assert isinstance(r.head_id, int)
        assert isinstance(r.tail_id, int)


def test_relation_ids_reference_valid_entities(rf):
    entity_ids = {e.id for e in rf.entities}
    for r in rf.relations[:100]:
        assert r.head_id in entity_ids, f"head_id {r.head_id} not in entities"
        assert r.tail_id in entity_ids, f"tail_id {r.tail_id} not in entities"


def test_name_to_id_mapping(rf):
    """name_to_id should map entity names to valid IDs."""
    assert len(rf.name_to_id) > 0
    for name, eid in list(rf.name_to_id.items())[:20]:
        assert eid in rf.id_to_entity


def test_id_to_entity_mapping(rf):
    """id_to_entity should map IDs back to Entity objects."""
    for eid, entity in list(rf.id_to_entity.items())[:20]:
        assert isinstance(entity, Entity)
        assert entity.id == eid


# ── 3. Quaternion Math ──────────────────────────────────────────


def test_hamilton_product_identity():
    """q * conjugate(q) should equal [1, 0, 0, 0] for unit quaternions."""
    q = np.array([0.5, 0.3, -0.2, 0.1])
    q = q / np.linalg.norm(q)  # normalize
    q_conj = quaternion_conjugate_np(q)
    result = hamilton_product_np(q, q_conj)
    np.testing.assert_allclose(result, [1, 0, 0, 0], atol=1e-6)


def test_hamilton_product_identity_random():
    """Identity property should hold for random unit quaternions."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        q = rng.standard_normal(4)
        q = q / np.linalg.norm(q)
        q_conj = quaternion_conjugate_np(q)
        result = hamilton_product_np(q, q_conj)
        np.testing.assert_allclose(result, [1, 0, 0, 0], atol=1e-6)


def test_conjugate_reversal():
    """conjugate(conjugate(q)) should equal q."""
    q = np.array([0.8, -0.3, 0.5, -0.1])
    double_conj = quaternion_conjugate_np(quaternion_conjugate_np(q))
    np.testing.assert_allclose(double_conj, q, atol=1e-10)


def test_conjugate_negates_imaginary():
    """Conjugate should negate imaginary parts, preserve real part."""
    q = np.array([0.5, 0.3, -0.2, 0.1])
    q_conj = quaternion_conjugate_np(q)
    assert q_conj[0] == q[0], "Real part should be preserved"
    np.testing.assert_allclose(q_conj[1:], -q[1:])


def test_hamilton_product_non_commutative():
    """Hamilton product is non-commutative: q1*q2 != q2*q1 in general."""
    q1 = np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2)
    q2 = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2)
    ab = hamilton_product_np(q1, q2)
    ba = hamilton_product_np(q2, q1)
    assert not np.allclose(ab, ba), "Hamilton product should be non-commutative"


def test_hamilton_product_batch():
    """Hamilton product should work on batched inputs."""
    batch = np.random.randn(10, 4)
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    batch = batch / norms
    conj = quaternion_conjugate_np(batch)
    result = hamilton_product_np(batch, conj)
    for i in range(10):
        np.testing.assert_allclose(result[i], [1, 0, 0, 0], atol=1e-5)


# ── 4. Trust Weight Bounds ──────────────────────────────────────


def test_trust_weights_bounded(rf):
    """All trust weights should be in [0, 1]."""
    assert np.all(rf.truth_weights >= 0.0), "Trust weights must be >= 0"
    assert np.all(rf.truth_weights <= 1.0), "Trust weights must be <= 1"


def test_trust_weights_not_uniform(rf):
    """Trust weights should show variation (not all identical)."""
    std = np.std(rf.truth_weights)
    assert std > 0.0, "Trust weights should not all be identical"


def test_trust_weights_finite(rf):
    """No NaN or Inf in trust weights."""
    assert np.all(np.isfinite(rf.truth_weights))


# ── 5. Stats ────────────────────────────────────────────────────


def test_get_stats(rf):
    stats = rf.get_stats()
    assert stats.entity_count == 4652
    assert stats.relation_count == 22289
    assert stats.embedding_dim == 16
    assert stats.file_size_mb > 0
    assert 0 <= stats.high_confidence_pct <= 100
    assert 0 <= stats.medium_confidence_pct <= 100
    assert 0 <= stats.low_confidence_pct <= 100
    # Percentages should sum to ~100
    total = stats.high_confidence_pct + stats.medium_confidence_pct + stats.low_confidence_pct
    assert abs(total - 100.0) < 0.1, f"Confidence percentages sum to {total}, expected ~100"


# ── 6. Audit ────────────────────────────────────────────────────


def test_audit_returns_result(rf):
    audit = rf.audit()
    assert hasattr(audit, "overall_score")
    assert hasattr(audit, "issues")
    assert hasattr(audit, "recommendations")


def test_audit_score_bounded(rf):
    audit = rf.audit()
    assert 0.0 <= audit.overall_score <= 1.0


def test_audit_issues_are_structured(rf):
    audit = rf.audit()
    for issue in audit.issues:
        assert hasattr(issue, "severity")
        assert hasattr(issue, "description")
        assert issue.severity in ("low", "medium", "high", "critical")


# ── 7. Claim Validation ────────────────────────────────────────


def test_validate_existing_claim(rf):
    """Validating a known relation should return a positive result."""
    r = rf.relations[0]
    head = rf.id_to_entity[r.head_id]
    tail = rf.id_to_entity[r.tail_id]
    rel_name = rf.relation_type_names.get(r.relation_type, str(r.relation_type))

    result = rf.validate_claim(head.name, rel_name, tail.name)
    assert result.truth_weight > 0, "Known relation should have positive trust weight"
    assert result.supporting_paths >= 0
    assert "NOT FOUND" not in result.verdict


def test_validate_nonexistent_claim(rf):
    """Validating a bogus relation should return zero confidence."""
    result = rf.validate_claim("ZZZZZ_FAKE", "has_nothing", "YYYYY_FAKE")
    assert result.truth_weight == 0.0
    assert result.supporting_paths == 0
    assert "UNKNOWN" in result.verdict or "NOT FOUND" in result.verdict


# ── 8. Navigation ──────────────────────────────────────────────


def test_get_relations_for_entity(rf):
    """get_relations should return relations for a known entity."""
    entity_name = rf.entities[0].name
    rels = rf.get_relations(entity_name)
    assert isinstance(rels, list)
    assert len(rels) > 0, f"Entity '{entity_name}' should have relations"


def test_get_top_hubs(rf):
    """Top hubs should return entities sorted by degree."""
    hubs = rf.get_top_hubs(5)
    assert len(hubs) <= 5
    assert len(hubs) > 0
    # Degrees should be descending
    degrees = [h.degree for h in hubs]
    assert degrees == sorted(degrees, reverse=True)


def test_list_entities_with_type_filter(rf):
    """list_entities should filter by type."""
    types = {e.entity_type for e in rf.entities}
    first_type = next(iter(types))
    filtered = rf.list_entities(type_filter=first_type, limit=10)
    assert len(filtered) > 0
    for e in filtered:
        assert e.entity_type == first_type


def test_list_entities_respects_limit(rf):
    result = rf.list_entities(limit=5)
    assert len(result) <= 5


# ── 9. Embeddings Integrity ────────────────────────────────────


def test_embeddings_finite(rf):
    """No NaN or Inf in embeddings."""
    assert np.all(np.isfinite(rf.embeddings))


def test_embeddings_not_zero(rf):
    """Embeddings should not be all zeros (training happened)."""
    norms = np.linalg.norm(rf.embeddings, axis=1)
    assert np.all(norms > 0), "No entity should have a zero embedding"
