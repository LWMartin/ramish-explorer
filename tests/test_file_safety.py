import os
"""
Test suite for v0.2 file safety validation.
"""
import struct
import tempfile
import pytest
from ramish_explorer.reader import RamishFile, MAX_ENTITY_COUNT, MAX_NAME_LENGTH

def test_reject_oversized_entity_count():
    """File claiming 10 billion entities should be rejected."""
    header = bytearray(64)
    header[0:6] = b'RAMISH'
    struct.pack_into('<H', header, 6, 1)       # version
    struct.pack_into('<I', header, 8, 0)       # flags
    struct.pack_into('<Q', header, 12, 10_000_000_000)  # absurd entity count
    struct.pack_into('<Q', header, 20, 0)      # relations
    struct.pack_into('<H', header, 28, 16)     # dim
    struct.pack_into('<H', header, 30, 1)      # rel types

    with tempfile.NamedTemporaryFile(suffix='.ramish', delete=False) as f:
        f.write(bytes(header))
        f.write(b'\x00' * 100)  # padding
        f.flush()
        with pytest.raises(ValueError, match="exceeds maximum"):
            RamishFile.load(f.name)

def test_reject_oversized_relation_count():
    """File claiming 10 billion relations should be rejected."""
    header = bytearray(64)
    header[0:6] = b'RAMISH'
    struct.pack_into('<H', header, 6, 1)
    struct.pack_into('<I', header, 8, 0)
    struct.pack_into('<Q', header, 12, 0)              # entities
    struct.pack_into('<Q', header, 20, 10_000_000_000)  # absurd relation count
    struct.pack_into('<H', header, 28, 16)
    struct.pack_into('<H', header, 30, 1)

    with tempfile.NamedTemporaryFile(suffix='.ramish', delete=False) as f:
        f.write(bytes(header))
        f.write(b'\x00' * 100)
        f.flush()
        with pytest.raises(ValueError, match="exceeds maximum"):
            RamishFile.load(f.name)

def test_reject_bad_magic():
    """File with bad magic bytes should be rejected."""
    header = bytearray(64)
    header[0:6] = b'BADMAG'

    with tempfile.NamedTemporaryFile(suffix='.ramish', delete=False) as f:
        f.write(bytes(header))
        f.write(b'\x00' * 100)
        f.flush()
        with pytest.raises(ValueError, match="bad magic"):
            RamishFile.load(f.name)

def test_valid_file_loads(rf_fixture_path=os.path.join(os.path.dirname(__file__), "fixtures", "chinook.ramish")):
    """A known valid file should load without safety errors."""
    rf = RamishFile.load(rf_fixture_path)
    assert len(rf.entities) > 0
