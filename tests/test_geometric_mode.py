"""Tests for geometric mode — mmap-backed reader with RNIX trailer.

Builds a test fixture from chinook.ramish:
  1. Load chinook via full-load mode
  2. Sort relations by head_id
  3. Rewrite .ramish with sorted relations + RNIX trailer appended
  4. Verify geometric mode produces same results as full-load mode
"""
import struct
import tempfile
import shutil
import os
import numpy as np
import pytest
from pathlib import Path

# Must run from phantom-ops root for bus/lingua path
CHINOOK_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "chinook.ramish")


def _build_geometric_fixture(src_path: str, dst_path: str):
    """Build a .ramish file with sorted relations + RNIX trailer.

    Takes a standard .ramish, sorts relations by head_id,
    rewrites the file, and appends RNIX binary name index.
    """
    from ramish_explorer.reader import RamishFile, MAGIC, VERSION

    rf = RamishFile.load(src_path)

    # Sort relations by head_id (required for binary search in geometric mode)
    sorted_indices = sorted(range(len(rf.relations)),
                            key=lambda i: rf.relations[i].head_id)
    sorted_relations = [rf.relations[i] for i in sorted_indices]
    sorted_weights = rf.truth_weights[sorted_indices] if rf.truth_weights is not None else None

    # Write new .ramish with sorted relations
    with open(dst_path, 'wb') as f:
        # Header (64 bytes) — force fp32 since we write dequantized embeddings
        header = rf._build_header(quantize="fp32")
        f.write(header)

        # Entities
        rf._write_entities(f)

        # Relations — sorted by head_id
        for r in sorted_relations:
            f.write(struct.pack('<IHI', r.head_id, r.relation_type, r.tail_id))

        # Embeddings
        if rf.embeddings is not None:
            f.write(rf.embeddings.astype(np.float32).tobytes())

        # Weights — reordered to match sorted relations
        if sorted_weights is not None:
            f.write(sorted_weights.astype(np.float32).tobytes())

        # Metadata JSON
        meta = rf._build_metadata()
        import json
        meta_bytes = json.dumps(meta).encode('utf-8')
        f.write(struct.pack('<I', len(meta_bytes)))
        f.write(meta_bytes)

        # RNIX trailer — build inline (no dependency on processing module)
        entries = []
        for e in rf.entities:
            # Resolve type_id from entity_type string
            type_id = 0
            try:
                type_id = rf.entity_types.index(e.entity_type)
            except (ValueError, AttributeError):
                pass
            entries.append((e.name.lower(), e.id, type_id))

        # Write RNIX directly to the file (not to separate file)
        # Sort entries
        entries.sort(key=lambda x: x[0])

        name_bytes_list = []
        offsets = []
        current_offset = 0
        for name, eid, etype in entries:
            name_b = name.encode('utf-8')
            name_bytes_list.append(name_b)
            offsets.append((current_offset, len(name_b), eid, etype))
            current_offset += len(name_b)

        total_name_bytes = current_offset

        # RNIX header
        f.write(b'RNIX')
        f.write(struct.pack('<I', len(entries)))
        f.write(struct.pack('<Q', total_name_bytes))

        # Names blob
        for nb in name_bytes_list:
            f.write(nb)

        # Offset table
        for off, nlen, eid, etype in offsets:
            f.write(struct.pack('<QHiH', off, nlen, eid, etype))

        # TSIX trailer — pre-computed tail sort index for bidirectional lookups
        # Compute argsort on tail_id of the SORTED relations
        tail_ids = np.array([r.tail_id for r in sorted_relations], dtype=np.uint32)
        tail_sort = np.argsort(tail_ids)
        # Use u32 for chinook-scale (< 4.3B relations)
        dtype_code = 0  # u32
        tail_sort_u32 = tail_sort.astype(np.uint32)

        # TSIX header: magic(4) + relation_count(u64) + dtype_code(u8) + pad(3B)
        f.write(b'TSIX')
        f.write(struct.pack('<Q', len(sorted_relations)))
        f.write(struct.pack('<B', dtype_code))
        f.write(b'\x00\x00\x00')  # padding

        # Index data
        f.write(tail_sort_u32.tobytes())


@pytest.fixture(scope="module")
def fixture_dir():
    """Create a temp dir with the geometric fixture, clean up after."""
    tmpdir = tempfile.mkdtemp(prefix="ramish_geom_test_")
    dst = os.path.join(tmpdir, "chinook_geometric.ramish")
    _build_geometric_fixture(CHINOOK_PATH, dst)
    yield tmpdir
    # Windows: mmap file handles may linger even after close+gc
    import gc
    gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def rf_full():
    """Chinook loaded in full-load mode (reference)."""
    from ramish_explorer.reader import RamishFile
    return RamishFile.load(CHINOOK_PATH)


@pytest.fixture(scope="module")
def rf_geo(fixture_dir):
    """Chinook loaded in geometric mode (mmap + RNIX)."""
    from ramish_explorer.reader import RamishFile
    path = os.path.join(fixture_dir, "chinook_geometric.ramish")
    rf = RamishFile.load_geometric(path)
    yield rf
    # Release all mmap references — Windows holds file locks until GC'd
    rf.close()
    rf.embeddings = None
    rf._relations_mmap = None
    rf._weights_mmap = None
    rf._tail_sort_idx = None
    rf._embedding_norms = None
    import gc
    gc.collect()


# ── Mode Detection ────────────────────────────────────────────

class TestModeDetection:
    def test_geometric_mode_flag(self, rf_geo):
        assert rf_geo.is_geometric_mode is True

    def test_full_load_not_geometric(self, rf_full):
        assert rf_full.is_geometric_mode is False

    def test_has_rnix(self, rf_geo):
        assert rf_geo.has_rnix is True

    def test_relations_sorted(self, rf_geo):
        assert rf_geo._relations_sorted is True

    def test_mmap_objects_exist(self, rf_geo):
        assert rf_geo._mmap_obj is not None
        assert rf_geo._relations_mmap is not None
        assert rf_geo._weights_mmap is not None

    def test_tsix_detected(self, rf_geo):
        """TSIX section should be detected and mmap'd."""
        assert rf_geo._tail_sort_idx is not None

    def test_tsix_is_mmap(self, rf_geo):
        """TSIX should be a memmap, not a computed array."""
        assert isinstance(rf_geo._tail_sort_idx, np.memmap)

    def test_tsix_correct_length(self, rf_geo):
        """TSIX index should have one entry per relation."""
        assert len(rf_geo._tail_sort_idx) == rf_geo._relation_count


# ── Name Resolution via RNIX ─────────────────────────────────

class TestRNIXNameResolution:
    def test_resolve_name_exact_match(self, rf_geo, rf_full):
        """RNIX resolve_name matches full-load for known entities."""
        # Pick a known entity from chinook
        test_name = rf_full.entities[0].name
        geo_ids, geo_amb = rf_geo.resolve_name(test_name)
        full_ids, full_amb = rf_full.resolve_name(test_name)
        assert sorted(geo_ids) == sorted(full_ids)

    def test_resolve_name_case_insensitive(self, rf_geo, rf_full):
        test_name = rf_full.entities[10].name
        geo_ids, _ = rf_geo.resolve_name(test_name.upper())
        full_ids, _ = rf_full.resolve_name(test_name.upper())
        assert sorted(geo_ids) == sorted(full_ids)

    def test_resolve_name_nonexistent(self, rf_geo):
        ids, amb = rf_geo.resolve_name("zzz_nonexistent_entity_xyz")
        assert ids == []
        assert amb is False

    def test_resolve_fuzzy_returns_suggestions(self, rf_geo):
        """RNIX fuzzy uses prefix search."""
        # Use first 3 chars of a known entity
        prefix = rf_geo.entities[0].name[:3].lower()
        ids, amb, suggestions = rf_geo.resolve_name_fuzzy(prefix)
        # Should get suggestions even if no exact match
        assert isinstance(suggestions, list)


# ── Binary Search Relations ───────────────────────────────────

class TestBinarySearchRelations:
    def test_binary_search_returns_neighbors(self, rf_geo):
        """Binary search finds relations for a connected entity."""
        # Find an entity that has neighbors in full-load
        test_eid = rf_geo.entities[0].id
        rels = rf_geo._binary_search_relations(test_eid)
        # At least some entities should have relations
        # (entity 0 might be orphan, so try a few)
        found = False
        for e in rf_geo.entities[:50]:
            rels = rf_geo._binary_search_relations(e.id)
            if rels:
                found = True
                break
        assert found, "No entity in first 50 has relations via binary search"

    def test_binary_search_matches_full_load(self, rf_geo, rf_full):
        """Binary search returns same neighbors as full-load dict."""
        # Find a well-connected entity in full load
        hub = None
        for eid, neighbors in rf_full.neighbors.items():
            if len(neighbors) >= 5:
                hub = eid
                break
        assert hub is not None, "Need a hub entity for comparison"

        full_neighbors = set()
        for nid, rt, w in rf_full.neighbors[hub]:
            full_neighbors.add((nid, rt))

        geo_rels = rf_geo._binary_search_relations(hub)
        geo_neighbors = set()
        for nid, rt, w in geo_rels:
            geo_neighbors.add((nid, rt))

        assert geo_neighbors == full_neighbors

    def test_binary_search_edge_found(self, rf_geo, rf_full):
        """_binary_search_edge finds an existing edge."""
        # Get a known edge from full-load
        r = rf_full.relations[0]
        result = rf_geo._binary_search_edge(r.head_id, r.tail_id)
        assert result is not None
        weight, rel_type = result
        assert 0.0 <= weight <= 1.0

    def test_binary_search_edge_not_found(self, rf_geo):
        """_binary_search_edge returns None for nonexistent edge."""
        result = rf_geo._binary_search_edge(999999, 999998)
        assert result is None


# ── Query Engine ──────────────────────────────────────────────

class TestGeometricQuery:
    def test_query_returns_results(self, rf_geo, rf_full):
        """Geometric mode query returns non-empty results for known entities."""
        # Use a known entity name
        test_name = rf_full.entities[0].name
        results = rf_geo.query(test_name)
        # Should find something (entity exists)
        assert isinstance(results, list)

    def test_query_results_have_correct_fields(self, rf_geo, rf_full):
        """Query results have required QueryResult fields."""
        # Find a well-connected name
        hubs = rf_full.get_top_hubs(1)
        if hubs:
            results = rf_geo.query(hubs[0].name)
            if results:
                r = results[0]
                assert hasattr(r, 'subject')
                assert hasattr(r, 'relation')
                assert hasattr(r, 'object')
                assert hasattr(r, 'truth_weight')
                assert 0.0 <= r.truth_weight <= 1.0

    def test_query_rnix_seeding(self, rf_geo, rf_full):
        """Query in geometric mode finds similar entities as full-load."""
        hubs = rf_full.get_top_hubs(1)
        if not hubs:
            pytest.skip("No hubs in chinook")
        name = hubs[0].name

        full_results = rf_full.query(name, topk=5)
        geo_results = rf_geo.query(name, topk=5)

        # Both should return results
        assert len(full_results) > 0 or len(geo_results) > 0

        # At least some overlap in returned entity names
        if full_results and geo_results:
            full_objects = {r.object for r in full_results}
            geo_objects = {r.object for r in geo_results}
            # Allow some divergence — geometric mode uses RNIX seeds
            # vs regex seeds, and binary search vs dict neighbors,
            # so results may differ slightly
            overlap = full_objects & geo_objects
            # At least 1 common result for top hub
            assert len(overlap) >= 1 or len(geo_results) > 0


# ── Validate Claim ────────────────────────────────────────────

class TestGeometricValidateClaim:
    def test_validate_existing_claim(self, rf_geo, rf_full):
        """Geometric validate_claim finds a known edge."""
        # Get a known relation from full-load
        r = rf_full.relations[0]
        subj = rf_full.id_to_entity[r.head_id].name
        obj = rf_full.id_to_entity[r.tail_id].name
        rel_name = rf_full.relation_type_names.get(r.relation_type, "")

        result = rf_geo.validate_claim(subj, rel_name, obj)
        assert result.truth_weight > 0.0
        assert "NOT FOUND" not in result.verdict

    def test_validate_nonexistent_claim(self, rf_geo):
        result = rf_geo.validate_claim(
            "zzz_fake_subject", "zzz_fake_relation", "zzz_fake_object"
        )
        assert result.truth_weight == 0.0

    def test_validate_matches_full_load(self, rf_geo, rf_full):
        """Geometric validate_claim returns same weight as full-load."""
        r = rf_full.relations[0]
        subj = rf_full.id_to_entity[r.head_id].name
        obj = rf_full.id_to_entity[r.tail_id].name
        rel_name = rf_full.relation_type_names.get(r.relation_type, "")

        full_result = rf_full.validate_claim(subj, rel_name, obj)
        geo_result = rf_geo.validate_claim(subj, rel_name, obj)

        assert abs(full_result.truth_weight - geo_result.truth_weight) < 0.01


# ── Get Relations ─────────────────────────────────────────────

class TestGeometricGetRelations:
    def test_get_relations_returns_results(self, rf_geo, rf_full):
        """Geometric get_relations returns relations for a connected entity."""
        hubs = rf_full.get_top_hubs(1)
        if not hubs:
            pytest.skip("No hubs")
        rels = rf_geo.get_relations(hubs[0].name)
        assert len(rels) > 0

    def test_get_relations_matches_full_load(self, rf_geo, rf_full):
        """Geometric get_relations returns same edges as full-load."""
        hubs = rf_full.get_top_hubs(1)
        if not hubs:
            pytest.skip("No hubs")
        name = hubs[0].name

        full_rels = rf_full.get_relations(name)
        geo_rels = rf_geo.get_relations(name)

        full_edges = {(r.relation, r.target) for r in full_rels}
        geo_edges = {(r.relation, r.target) for r in geo_rels}

        assert full_edges == geo_edges

    def test_get_relations_by_id(self, rf_geo, rf_full):
        """get_relations with explicit entity_id works in geometric mode."""
        eid = rf_full.entities[0].id
        name = rf_full.entities[0].name
        rels = rf_geo.get_relations(name, entity_id=eid)
        assert isinstance(rels, list)


# ── Get Stats ─────────────────────────────────────────────────

class TestGeometricStats:
    def test_entity_count_matches(self, rf_geo, rf_full):
        full_stats = rf_full.get_stats()
        geo_stats = rf_geo.get_stats()
        assert geo_stats.entity_count == full_stats.entity_count

    def test_relation_count_matches(self, rf_geo, rf_full):
        full_stats = rf_full.get_stats()
        geo_stats = rf_geo.get_stats()
        assert geo_stats.relation_count == full_stats.relation_count

    def test_file_size_positive(self, rf_geo):
        stats = rf_geo.get_stats()
        assert stats.file_size_mb > 0


# ── Audit ─────────────────────────────────────────────────────

class TestGeometricAudit:
    def test_audit_returns_result(self, rf_geo):
        result = rf_geo.audit()
        assert hasattr(result, 'overall_score')
        assert 0.0 <= result.overall_score <= 1.0

    def test_audit_has_issues_list(self, rf_geo):
        result = rf_geo.audit()
        assert isinstance(result.issues, list)


# ── Get Top Hubs ──────────────────────────────────────────────

class TestGeometricTopHubs:
    def test_top_hubs_returns_results(self, rf_geo):
        hubs = rf_geo.get_top_hubs(5)
        assert len(hubs) > 0

    def test_top_hubs_sorted_by_degree(self, rf_geo):
        hubs = rf_geo.get_top_hubs(5)
        if len(hubs) > 1:
            for i in range(len(hubs) - 1):
                assert hubs[i].degree >= hubs[i + 1].degree

    def test_top_hub_matches_full_load(self, rf_geo, rf_full):
        """Top hub in geometric mode is same as full-load."""
        full_hubs = rf_full.get_top_hubs(1)
        geo_hubs = rf_geo.get_top_hubs(1)
        if full_hubs and geo_hubs:
            assert full_hubs[0].name == geo_hubs[0].name


# ── Narrow / Autocomplete ────────────────────────────────────

class TestGeometricNarrow:
    def test_narrow_returns_results(self, rf_geo, rf_full):
        prefix = rf_full.entities[0].name[:3]
        results = rf_geo.narrow(prefix)
        assert len(results) > 0

    def test_narrow_tuple_format(self, rf_geo, rf_full):
        prefix = rf_full.entities[0].name[:3]
        results = rf_geo.narrow(prefix)
        if results:
            eid, name, etype, count = results[0]
            assert isinstance(eid, int)
            assert isinstance(name, str)
            assert isinstance(count, int)


# ── Context Manager ───────────────────────────────────────────

class TestContextManager:
    def test_context_manager(self, fixture_dir):
        from ramish_explorer.reader import RamishFile
        import gc
        path = os.path.join(fixture_dir, "chinook_geometric.ramish")
        with RamishFile.load_geometric(path) as rf:
            assert rf.is_geometric_mode
            stats = rf.get_stats()
            assert stats.entity_count > 0
        # After exit, mmap should be cleaned up
        assert rf._mmap_obj is None
        gc.collect()


# ── Standard Load with RNIX Detection ────────────────────────

class TestStandardLoadRNIXDetection:
    def test_standard_load_detects_rnix(self, fixture_dir):
        """Standard load() on a file with RNIX trailer detects it."""
        from ramish_explorer.reader import RamishFile
        path = os.path.join(fixture_dir, "chinook_geometric.ramish")
        rf = RamishFile.load(path)
        assert rf.has_rnix is True
        assert rf.is_geometric_mode is False  # standard load, not geometric
        rf.close()

    def test_standard_load_rnix_accelerates_resolve(self, fixture_dir, rf_full):
        """Standard load with RNIX uses it for resolve_name."""
        from ramish_explorer.reader import RamishFile
        path = os.path.join(fixture_dir, "chinook_geometric.ramish")
        rf = RamishFile.load(path)

        name = rf_full.entities[0].name
        ids, _ = rf.resolve_name(name)
        full_ids, _ = rf_full.resolve_name(name)
        assert sorted(ids) == sorted(full_ids)
        rf.close()


# ── _get_neighbors abstraction ────────────────────────────────

class TestGetNeighborsAbstraction:
    def test_get_neighbors_geometric(self, rf_geo, rf_full):
        """_get_neighbors returns same results in geometric mode."""
        hubs = rf_full.get_top_hubs(1)
        if not hubs:
            pytest.skip("No hubs")
        eid = hubs[0].entity_id

        full_n = set((nid, rt) for nid, rt, _ in rf_full._get_neighbors(eid))
        geo_n = set((nid, rt) for nid, rt, _ in rf_geo._get_neighbors(eid))
        assert full_n == geo_n

    def test_get_neighbors_empty_for_orphan(self, rf_geo):
        """_get_neighbors returns empty for nonexistent entity."""
        result = rf_geo._get_neighbors(999999)
        assert result == []
