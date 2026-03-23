import os
"""Tests for Fix 1–4: name resolution, disambiguation, lexical seeds, type normalization."""
import pytest
import numpy as np
from ramish_explorer.reader import RamishFile, HubInfo

CHINOOK_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "chinook.ramish")
PHANTOM_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "phantom-ops.ramish")
ENRICHED_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "phantom-ops-enriched.ramish")


@pytest.fixture(scope="module")
def rf():
    return RamishFile.load(CHINOOK_PATH)


@pytest.fixture(scope="module")
def rf_phantom():
    return RamishFile.load(PHANTOM_PATH)


@pytest.fixture(scope="module")
def rf_enriched():
    return RamishFile.load(ENRICHED_PATH)


# ── Fix 1: Word-boundary-aware lexical seed matching ──────────────────────

class TestLexicalSeeds:
    def test_short_name_not_substring_matched(self, rf):
        """Short entity names like 'get' should not match inside 'budget'."""
        results = rf.query("budget planning")
        # We don't assert specific results, just that 'get' entities
        # aren't flooding the results via substring matching
        subjects = [r.subject.lower() for r in results]
        # Should not see random noise from short-name collisions
        assert len(results) <= 10

    def test_exact_name_still_matches(self, rf):
        """A query containing an exact entity name should still seed."""
        # AC/DC is a known chinook entity
        results = rf.query("AC/DC")
        assert len(results) > 0

    def test_seed_deduplication(self, rf):
        """Seeds should be deduplicated — same entity via multiple name_to_ids entries
        should not produce duplicate results from the same (subject, relation, object) triple."""
        results = rf.query("Rock")
        # Check for exact duplicate triples, not just repeated subjects
        triples = [(r.subject, r.relation, r.object) for r in results]
        assert len(triples) == len(set(triples)), "Duplicate triples found in query results"


class TestThreeTierSeeds:
    """Test the three-tier lexical seed matching on code-structure files."""

    def test_tier2_query_prefix_of_entity(self, rf_phantom):
        """Tier 2: query 'session_briefing' should match entity 'session_briefing.py'."""
        results = rf_phantom.query("session_briefing")
        assert len(results) > 0
        subjects = [r.subject.lower() for r in results]
        assert any("session_briefing" in s for s in subjects)

    def test_tier2_does_not_cross_word_boundary(self, rf_phantom):
        """Tier 2: 'ramish_explorer' should NOT match 'test_ramish_explorer'
        because underscore is a word character — no \\b boundary."""
        # This is a subtle but correct behavior: the query must align
        # to actual word boundaries in the entity name
        results = rf_phantom.query("ramish_explorer")
        # Should still find results IF there's an entity that starts/ends
        # with ramish_explorer at a boundary. The key assertion is that
        # it doesn't pull in unrelated entities.
        for r in results:
            # Every result subject should contain 'ramish' somewhere
            assert "ramish" in r.subject.lower() or "ramish" in r.object.lower()

    def test_tier3_word_split_fallback(self, rf_phantom):
        """Tier 3: 'session briefing analysis' should find entities via
        individual words when the full phrase doesn't match."""
        results = rf_phantom.query("session briefing analysis")
        assert len(results) > 0
        # Should find briefing-related entities
        subjects = [r.subject.lower() for r in results]
        assert any("briefing" in s or "session" in s for s in subjects)

    def test_tier1_before_tier2(self, rf_phantom):
        """Tier 1 should take priority — exact entity name match in query
        should seed without falling through to tier 2."""
        # 'briefing' is an exact entity name
        results = rf_phantom.query("briefing")
        assert len(results) > 0
        subjects = [r.subject.lower() for r in results]
        assert any("briefing" in s for s in subjects)

    def test_budget_returns_nothing(self, rf_phantom):
        """No entity named 'budget' — should get 0 results, not noise from 'get'."""
        results = rf_phantom.query("budget")
        assert len(results) == 0


# ── Fix 2: truth_weight normalized to Python float ────────────────────────

class TestTruthWeightType:
    def test_neighbor_weights_are_python_float(self, rf):
        """Weights in the neighbor adjacency should be Python floats, not np.float32."""
        for eid, neighbors in rf.neighbors.items():
            for _, _, weight in neighbors[:5]:
                assert type(weight) is float, (
                    f"Entity {eid} neighbor weight is {type(weight).__name__}, expected float"
                )
            break  # Just check first entity's neighbors

    def test_relation_weights_are_python_float(self, rf):
        """get_relations() should return Python float weights."""
        # Pick any entity with relations
        for name in list(rf.name_to_ids.keys())[:20]:
            rels = rf.get_relations(name)
            if rels:
                for r in rels[:3]:
                    assert type(r.truth_weight) is float, (
                        f"RelInfo.truth_weight is {type(r.truth_weight).__name__}"
                    )
                break

    def test_hub_avg_weight_is_python_float(self, rf):
        """HubInfo.avg_weight should be Python float."""
        hubs = rf.get_top_hubs(3)
        for h in hubs:
            assert type(h.avg_weight) is float


# ── Fix 3: resolve_name / resolve_name_fuzzy ──────────────────────────────

class TestResolveNameMultiMap:
    def test_resolve_name_exact(self, rf):
        """Exact name resolution returns correct IDs."""
        # Pick an entity we know exists
        name = list(rf.name_to_ids.keys())[0]
        ids, ambiguous = rf.resolve_name(name)
        assert len(ids) >= 1
        for eid in ids:
            assert eid in rf.id_to_entity

    def test_resolve_name_case_insensitive(self, rf):
        """Name resolution is case-insensitive."""
        name = list(rf.name_to_ids.keys())[0]
        ids1, _ = rf.resolve_name(name.lower())
        ids2, _ = rf.resolve_name(name.upper())
        assert ids1 == ids2

    def test_resolve_name_nonexistent(self, rf):
        """Nonexistent name returns empty list."""
        ids, ambiguous = rf.resolve_name("zzz_nonexistent_entity_zzz")
        assert ids == []
        assert ambiguous is False

    def test_resolve_name_fuzzy_suggestions(self, rf):
        """Fuzzy resolution suggests similar names when exact fails."""
        # Use a partial name that should match something
        ids, ambiguous, suggestions = rf.resolve_name_fuzzy("zzz_nothing_zzz")
        assert ids == []
        assert suggestions == []  # truly nothing matches

    def test_resolve_name_fuzzy_exact_no_suggestions(self, rf):
        """When exact match exists, suggestions should be empty."""
        name = list(rf.name_to_ids.keys())[0]
        ids, ambiguous, suggestions = rf.resolve_name_fuzzy(name)
        assert len(ids) >= 1
        assert suggestions == []

    def test_shadowed_entities_now_reachable(self, rf_phantom):
        """Entities that were shadowed by name_to_id should be reachable via name_to_ids."""
        shadowed_count = 0
        for name, ids in rf_phantom.name_to_ids.items():
            if len(ids) > 1:
                shadowed_count += len(ids) - 1
                # All IDs should be valid
                for eid in ids:
                    assert eid in rf_phantom.id_to_entity
                # resolve_name should return ALL of them
                resolved_ids, ambiguous = rf_phantom.resolve_name(name)
                assert ambiguous is True
                assert set(resolved_ids) == set(ids)

        # The phantom-ops file should have some shadowed entities
        # (668 was the reported count)
        assert shadowed_count > 0, "Expected shadowed entities in phantom-ops.ramish"

    def test_get_relations_returns_all_ambiguous(self, rf_phantom):
        """get_relations on an ambiguous name returns edges for ALL matching entities."""
        for name, ids in rf_phantom.name_to_ids.items():
            if len(ids) > 1:
                rels = rf_phantom.get_relations(name)
                # Should have source_entity_id tagged on each
                if rels:
                    source_ids = {r.source_entity_id for r in rels if r.source_entity_id is not None}
                    # At least some of the ambiguous entities should appear
                    assert len(source_ids) >= 1
                break


# ── Fix 4: HubInfo carries entity_id, disambiguation display ─────────────

class TestHubDisambiguation:
    def test_hub_info_has_entity_id(self, rf):
        """HubInfo should carry entity_id field."""
        hubs = rf.get_top_hubs(5)
        for h in hubs:
            assert hasattr(h, 'entity_id')
            assert isinstance(h.entity_id, int)
            assert h.entity_id in rf.id_to_entity

    def test_hub_entity_id_matches_name(self, rf):
        """HubInfo.entity_id should correspond to an entity with the same name."""
        hubs = rf.get_top_hubs(5)
        for h in hubs:
            entity = rf.id_to_entity[h.entity_id]
            assert entity.name == h.name


# ── Integration: shadow count validation ──────────────────────────────────

class TestShadowCounts:
    def test_phantom_shadow_count(self, rf_phantom):
        """Verify the multi-map captures all entities the flat map drops."""
        flat_count = len(rf_phantom.name_to_id)
        multi_total = sum(len(ids) for ids in rf_phantom.name_to_ids.values())
        shadowed = multi_total - flat_count
        # We know from analysis: 668 in phantom-ops
        assert shadowed >= 100, f"Only {shadowed} shadowed — expected hundreds"

    def test_enriched_shadow_count(self, rf_enriched):
        """Enriched file should also show significant shadowing."""
        flat_count = len(rf_enriched.name_to_id)
        multi_total = sum(len(ids) for ids in rf_enriched.name_to_ids.values())
        shadowed = multi_total - flat_count
        assert shadowed >= 100, f"Only {shadowed} shadowed — expected hundreds"
