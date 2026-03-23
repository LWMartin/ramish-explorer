"""
.ramish File Format Reader — v0.2.0

Binary format reader for portable knowledge graphs with truth weights.
This is the read-only version for the ramish-explorer package.

v0.2 changes:
- Canonical embedding_to_quats() helper (Step 2)
- Relation-type and compound indexes (Step 3)
- Precomputed embedding norms (Step 4)
- Sign-aligned frozen key extraction with lazy cache (Step 5)
- File safety validation (Step 7)
- Hybrid geometric query engine with graph rerank (Step 8)
"""
import struct
import json
import re
import mmap
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .models import Entity, Relation
from .quantize import EmbeddingQuantizer
from .quate import hamilton_product_np, quaternion_conjugate_np


MAGIC = b'RAMISH'
RNIX_MAGIC = b'RNIX'
TSIX_MAGIC = b'TSIX'  # Tail Sort IndeX — pre-computed argsort on tail_id
VERSION = 1

# Quantization type codes stored in header flags field
QUANT_FP32 = 0
QUANT_FP16 = 1
QUANT_INT8 = 2
QUANT_DTYPE_MAP = {QUANT_FP32: "fp32", QUANT_FP16: "fp16", QUANT_INT8: "int8"}
QUANT_CODE_MAP = {"fp32": QUANT_FP32, "fp16": QUANT_FP16, "int8": QUANT_INT8}

# v0.2: Safety limits
MAX_ENTITY_COUNT = 500_000_000       # 500M entities
MAX_RELATION_COUNT = 5_000_000_000   # 5B relations
MAX_NAME_LENGTH = 10_000             # 10K bytes per entity name


@dataclass
class QueryResult:
    """A query result with truth weight."""
    subject: str
    relation: str
    object: str
    truth_weight: float
    supporting_paths: int


@dataclass
class ValidationResult:
    """Result of validating a specific claim."""
    truth_weight: float
    supporting_paths: int
    verdict: str


@dataclass
class RamishStats:
    """Statistics about a .ramish file."""
    entity_count: int
    relation_count: int
    embedding_dim: int
    file_size_mb: float
    compression_ratio: float
    high_confidence_pct: float
    medium_confidence_pct: float
    low_confidence_pct: float


@dataclass
class AuditIssue:
    """A data quality issue."""
    severity: str
    description: str
    affected_count: int


@dataclass
class AuditResult:
    """Result of a data quality audit."""
    overall_score: float
    issues: List[AuditIssue]
    recommendations: List[str]


@dataclass
class HubInfo:
    """Information about a hub entity."""
    entity_id: int
    name: str
    entity_type: str
    degree: int
    thick_cables: int
    loose_threads: int
    avg_weight: float


def get_verdict(weight: float, num_paths: int) -> str:
    """Determine confidence verdict based on weight and path count."""
    if weight > 0.8 and num_paths > 500:
        return "VERY HIGH CONFIDENCE - overwhelming structural support"
    elif weight > 0.7 and num_paths > 100:
        return "HIGH CONFIDENCE - strong verification"
    elif weight > 0.4 and num_paths > 50:
        return "MODERATE CONFIDENCE - some supporting evidence"
    else:
        return "LOW CONFIDENCE - limited verification"


# Relation section record layout (10 bytes): head_id(u32) + rel_type(u16) + tail_id(u32)
RELATION_DTYPE = np.dtype([('head_id', '<u4'), ('rel_type', '<u2'), ('tail_id', '<u4')])
RNIX_HEADER_SIZE = 16
RNIX_ENTRY_SIZE = 16  # u64 + u16 + i32 + u2


class _RNIXReader:
    """Read-only mmap-backed RNIX name index. Zero RAM for lookups."""

    def __init__(self, mm: mmap.mmap, offset: int):
        """Initialize from an existing mmap at a given byte offset."""
        self._mm = mm
        self._base = offset

        assert mm[offset:offset + 4] == RNIX_MAGIC, f"Bad RNIX magic at offset {offset}"
        self.entry_count = struct.unpack_from('<I', mm, offset + 4)[0]
        self._names_blob_size = struct.unpack_from('<Q', mm, offset + 8)[0]
        self._names_start = offset + RNIX_HEADER_SIZE
        self._table_start = self._names_start + self._names_blob_size

    def _get_name(self, idx: int) -> bytes:
        pos = self._table_start + idx * RNIX_ENTRY_SIZE
        off = struct.unpack_from('<Q', self._mm, pos)[0]
        nlen = struct.unpack_from('<H', self._mm, pos + 8)[0]
        start = self._names_start + off
        return self._mm[start:start + nlen]

    def _get_entry(self, idx: int) -> Tuple[bytes, int, int]:
        pos = self._table_start + idx * RNIX_ENTRY_SIZE
        off, nlen, eid, etype = struct.unpack_from('<QHiH', self._mm, pos)
        start = self._names_start + off
        name = self._mm[start:start + nlen]
        return name, eid, etype

    def search(self, query: str, max_results: int = 20) -> List[Tuple[int, int, str, str]]:
        """Binary search for exact and prefix matches.

        Returns list of (entity_id, type_id, name, match_type)
        where match_type is 'exact' or 'prefix'.
        """
        q = query.lower().encode('utf-8')
        lo, hi = 0, self.entry_count
        while lo < hi:
            mid = (lo + hi) // 2
            if self._get_name(mid) < q:
                lo = mid + 1
            else:
                hi = mid

        results = []
        i = lo
        while i < self.entry_count and len(results) < max_results:
            name, eid, etype = self._get_entry(i)
            if name == q:
                results.append((eid, etype, name.decode('utf-8'), 'exact'))
            elif name.startswith(q):
                results.append((eid, etype, name.decode('utf-8'), 'prefix'))
            else:
                break
            i += 1
        return results

    def resolve_ids(self, query: str) -> List[int]:
        """Resolve a name to entity IDs (exact matches only)."""
        results = self.search(query)
        return [eid for eid, _, _, mtype in results if mtype == 'exact']


class RamishFile:
    """Handler for .ramish file format (read-only)."""

    def __init__(self, path: Optional[str] = None):
        """Initialize a RamishFile, optionally loading from a .ramish file.

        Usage:
            rf = RamishFile("data.ramish")       # load on construction
            rf = RamishFile.load("data.ramish")   # classmethod (equivalent)
            rf = RamishFile()                      # empty instance (advanced use)
        """
        self.entities: List[Any] = []
        self.relations: List[Any] = []
        self.relation_types: Dict[str, int] = {}
        self.relation_type_names: Dict[int, str] = {}
        self.entity_types: List[str] = []

        self.embeddings: Optional[np.ndarray] = None
        self.truth_weights: Optional[np.ndarray] = None

        self.name_to_id: Dict[str, int] = {}
        self.name_to_ids: Dict[str, List[int]] = {}  # multi-map for duplicate names
        self.id_to_entity: Dict[int, Any] = {}

        self.neighbors: Dict[int, List[Tuple[int, int, float]]] = {}

        self._avg_weight: Optional[float] = None
        self._weight_threshold: Optional[float] = None

        self._loaded_path: Optional[Path] = None
        self._quantize_dtype: str = "fp32"

        # v0.2 Step 3: Relation indexes
        self.edges_by_rel_type: Dict[int, List[int]] = {}
        self.out_edges: Dict[int, Dict[int, List[int]]] = {}
        self.in_edges: Dict[int, Dict[int, List[int]]] = {}
        self.targets_by_head_rel: Dict[Tuple[int, int], List[int]] = {}
        self.heads_by_tail_rel: Dict[Tuple[int, int], List[int]] = {}

        # v0.2 Step 4: Precomputed norms
        self._embedding_norms: Optional[np.ndarray] = None

        # v0.2 Step 5: Frozen key cache
        self._frozen_keys: Dict[int, np.ndarray] = {}
        self._frozen_key_stability: Dict[int, float] = {}

        # Progressive Name Index (PNI) — optional, for scale-ready name resolution
        self._pni = None

        # Geometric mode — mmap-backed, zero-RAM for large files
        self._geometric_mode: bool = False
        self._rnix: Optional[_RNIXReader] = None
        self._mmap_file = None
        self._mmap_obj: Optional[mmap.mmap] = None
        self._relations_mmap: Optional[np.ndarray] = None
        self._weights_mmap: Optional[np.ndarray] = None
        self._relation_section_offset: int = 0
        self._embedding_section_offset: int = 0
        self._weight_section_offset: int = 0
        self._relations_sorted: bool = False
        self._tail_sort_idx: Optional[np.ndarray] = None

        if path is not None:
            self._load_from_path(path)

    # ── Step 2: Canonical quaternion layout helper ─────────────────

    def embedding_to_quats(self, entity_ids: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Extract quaternion arrays from stored embeddings.

        Stored layout (component-major): [a1..ad, b1..bd, c1..cd, d1..dd]
        Output: (n, dim, 4) array where last axis is [a, b, c, d] per dimension.

        Single entity: returns (dim, 4)
        Multiple entities: returns (n, dim, 4)
        """
        if isinstance(entity_ids, int):
            emb = self.embeddings[entity_ids]  # (dim*4,)
            return emb.reshape(4, self._embedding_dim).T  # (dim, 4)
        else:
            ids = np.asarray(entity_ids)
            emb = self.embeddings[ids]  # (n, dim*4)
            return emb.reshape(len(ids), 4, self._embedding_dim).transpose(0, 2, 1)  # (n, dim, 4)

    # ── Step 5: Sign-aligned frozen key extraction ────────────────

    def _extract_frozen_key(self, rel_type: int) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract sign-aligned frozen key for a relation type.

        Returns:
            (key, stability) where key is (dim, 4) and stability is [0, 1].
            Returns (None, 0.0) if no instances exist.
        """
        # Check cache first
        if rel_type in self._frozen_keys:
            return self._frozen_keys[rel_type], self._frozen_key_stability[rel_type]

        indices = self.edges_by_rel_type.get(rel_type, [])
        if not indices:
            return None, 0.0

        head_ids = [self.relations[i].head_id for i in indices]
        tail_ids = [self.relations[i].tail_id for i in indices]

        h_q = self.embedding_to_quats(head_ids)  # (n, dim, 4)
        t_q = self.embedding_to_quats(tail_ids)  # (n, dim, 4)

        # Individual keys: conj(head) * tail
        individual_keys = hamilton_product_np(quaternion_conjugate_np(h_q), t_q)  # (n, dim, 4)

        # Sign alignment: align each key to the first one
        if len(individual_keys) > 1:
            reference = individual_keys[0]  # (dim, 4)
            # For each key, check if dot product with reference is negative
            # If so, flip the sign (q and -q represent the same rotation)
            for i in range(1, len(individual_keys)):
                dot = np.sum(individual_keys[i] * reference)
                if dot < 0:
                    individual_keys[i] = -individual_keys[i]

        frozen_key = np.mean(individual_keys, axis=0)  # (dim, 4)

        # Stability: how consistent are individual keys vs the mean?
        deviations = np.linalg.norm(individual_keys - frozen_key, axis=-1)  # (n, dim)
        avg_dev = float(np.mean(deviations))
        key_norm = float(np.mean(np.linalg.norm(frozen_key, axis=-1)))
        stability = max(0.0, 1.0 - avg_dev / max(key_norm, 1e-8))

        # Cache it
        self._frozen_keys[rel_type] = frozen_key
        self._frozen_key_stability[rel_type] = stability

        return frozen_key, stability

    # ── Load / Save ───────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> 'RamishFile':
        """Load from .ramish file (full mode — all data in RAM).

        Shorthand for ``RamishFile(path)``.  Both patterns work::

            rf = RamishFile.load("data.ramish")   # classmethod
            rf = RamishFile("data.ramish")         # constructor
        """
        return cls(path)

    @classmethod
    def load_geometric(cls, path: str) -> 'RamishFile':
        """Load in geometric mode — mmap-backed for large files.

        Entities are read into memory for display names.
        Embeddings, relations, and weights are memory-mapped (zero RAM).
        Name resolution uses RNIX binary search instead of dicts.
        Relation lookups use binary search on sorted relation section.

        Requires RNIX trailer (written by the assembler).
        Use this when standard load() would exhaust RAM.

        Usage::

            rf = RamishFile.load_geometric("science_core.ramish")
            results = rf.query("quantum computing")
        """
        instance = cls.__new__(cls)
        instance.__init__()
        instance._load_geometric_from_path(path)
        return instance

    def _load_geometric_from_path(self, path: str):
        """Internal: load in geometric mode with mmap-backed access."""
        path = Path(path)
        self._loaded_path = path
        self._geometric_mode = True

        file_size = path.stat().st_size

        with open(path, 'rb') as f:
            header = f.read(64)
            self._parse_header(header)
            self._validate_header(header, file_size)

            # Read entity names (needed for display)
            self._read_entities(f)

            # Record relation section offset, SKIP reading into Python list
            self._relation_section_offset = f.tell()
            relation_section_bytes = self._relation_count * 10
            f.seek(relation_section_bytes, 1)

            # Record embedding section offset, SKIP reading into numpy
            self._embedding_section_offset = f.tell()
            if self._entity_count > 0 and self._embedding_dim > 0:
                total_components = self._entity_count * self._embedding_dim * 4
                quant_code = self._quant_code
                bytes_per = {QUANT_FP16: 2, QUANT_INT8: 1}.get(quant_code, 4)
                f.seek(total_components * bytes_per, 1)
                self._quantize_dtype = QUANT_DTYPE_MAP.get(quant_code, "fp32")

            # Record weight section offset, skip
            self._weight_section_offset = f.tell()
            f.seek(self._relation_count * 4, 1)

            # Read metadata
            meta_len = struct.unpack('<I', f.read(4))[0]
            meta_bytes = f.read(meta_len)
            self._parse_metadata(json.loads(meta_bytes.decode('utf-8')))

            # RNIX trailer required
            self._rnix_offset = f.tell()
            trailer_magic = f.read(4)
            if trailer_magic != RNIX_MAGIC:
                raise ValueError(
                    "Geometric mode requires RNIX trailer. "
                    "Use RamishFile.load() for files without RNIX."
                )

            # Compute end of RNIX section to check for TSIX
            rnix_entry_count = struct.unpack_from('<I', f.read(4), 0)[0]
            rnix_names_blob_size = struct.unpack_from('<Q', f.read(8), 0)[0]
            rnix_total = (RNIX_HEADER_SIZE +
                          rnix_names_blob_size +
                          rnix_entry_count * RNIX_ENTRY_SIZE)
            self._tsix_offset = None
            tsix_check_pos = self._rnix_offset + rnix_total
            f.seek(tsix_check_pos)
            tsix_magic = f.read(4)
            if tsix_magic == TSIX_MAGIC:
                self._tsix_offset = tsix_check_pos

        # mmap the entire file
        self._mmap_file = open(path, 'rb')
        self._mmap_obj = mmap.mmap(
            self._mmap_file.fileno(), 0, access=mmap.ACCESS_READ
        )

        # RNIX for name resolution
        self._rnix = _RNIXReader(self._mmap_obj, self._rnix_offset)

        # mmap relations as structured numpy array
        self._relations_mmap = np.memmap(
            path, dtype=RELATION_DTYPE, mode='r',
            offset=self._relation_section_offset,
            shape=(self._relation_count,)
        )
        # Check if sorted by head_id
        if self._relation_count > 1:
            sample = min(1000, self._relation_count)
            heads = self._relations_mmap['head_id'][:sample]
            self._relations_sorted = bool(np.all(heads[:-1] <= heads[1:]))

        # Tail sort index for bidirectional lookups
        # Priority 1: TSIX section (pre-computed at assembly time, mmap'd — zero RAM)
        # Priority 2: Compute on load for small files (< 50M relations)
        # Priority 3: None — incoming lookups degrade gracefully (outgoing only)
        self._tail_sort_idx = None
        if self._tsix_offset is not None and self._relations_sorted:
            # TSIX format: magic(4B) + relation_count(u64, 8B) + dtype_code(u8) + pad(3B) = 16B header
            # Then: relation_count × sizeof(dtype) indices
            tsix_header_size = 16
            tsix_dtype_code = self._mmap_obj[self._tsix_offset + 12]
            tsix_dtype = np.uint64 if tsix_dtype_code == 1 else np.uint32
            self._tail_sort_idx = np.memmap(
                path, dtype=tsix_dtype, mode='r',
                offset=self._tsix_offset + tsix_header_size,
                shape=(self._relation_count,)
            )
        elif self._relations_sorted and self._relation_count <= 50_000_000:
            # Small file fallback: compute on load (~400 MB for 50M rels)
            tails = self._relations_mmap['tail_id']
            self._tail_sort_idx = np.argsort(tails).astype(np.int64)

        # mmap embeddings (keep raw dtype — dequantize on access for queries)
        if self._entity_count > 0 and self._embedding_dim > 0:
            embed_dtype_map = {QUANT_FP16: np.float16, QUANT_INT8: np.int8}
            embed_dtype = embed_dtype_map.get(self._quant_code, np.float32)
            self.embeddings = np.memmap(
                path, dtype=embed_dtype, mode='r',
                offset=self._embedding_section_offset,
                shape=(self._entity_count, self._embedding_dim * 4)
            )
            # Norms for cosine similarity (reads through mmap, stores only 1D array)
            emb_f32 = self.embeddings.astype(np.float32) if embed_dtype != np.float32 else self.embeddings
            self._embedding_norms = np.linalg.norm(emb_f32, axis=1)

        # mmap trust weights
        if self._relation_count > 0:
            self._weights_mmap = np.memmap(
                path, dtype=np.float32, mode='r',
                offset=self._weight_section_offset,
                shape=(self._relation_count,)
            )

        # Minimal indexes (entities only — no relation dicts)
        self._build_indexes_geometric()

    def _load_from_path(self, path: str):
        """Internal: populate this instance from a .ramish file.

        Handles both legacy (fp32) and quantized (fp16/int8) formats.
        Embeddings are always dequantized to fp32 on load so all
        downstream code works without modification.
        Detects RNIX trailer for name resolution acceleration.
        """
        path = Path(path)
        self._loaded_path = path

        file_size = path.stat().st_size

        with open(path, 'rb') as f:
            header = f.read(64)
            self._parse_header(header)

            # v0.2 Step 7: Validate header before allocating
            self._validate_header(header, file_size)

            self._read_entities(f)

            self._relation_section_offset = f.tell()
            self._read_relations(f)

            self._embedding_section_offset = f.tell()

            if self._entity_count > 0 and self._embedding_dim > 0:
                total_components = self._entity_count * self._embedding_dim * 4
                quant_code = self._quant_code

                if quant_code == QUANT_FP16:
                    embed_bytes = f.read(total_components * 2)
                    raw = np.frombuffer(embed_bytes, dtype=np.float16).reshape(
                        self._entity_count, self._embedding_dim * 4
                    )
                    self.embeddings = EmbeddingQuantizer.dequantize({
                        "embeddings": raw, "dtype": "fp16"
                    })
                    self._quantize_dtype = "fp16"
                elif quant_code == QUANT_INT8:
                    embed_bytes = f.read(total_components)
                    raw = np.frombuffer(embed_bytes, dtype=np.int8).reshape(
                        self._entity_count, self._embedding_dim * 4
                    )
                    self.embeddings = EmbeddingQuantizer.dequantize({
                        "embeddings": raw, "dtype": "int8", "scale": self._int8_scale
                    })
                    self._quantize_dtype = "int8"
                else:
                    embed_bytes = f.read(total_components * 4)
                    self.embeddings = np.frombuffer(embed_bytes, dtype=np.float32).reshape(
                        self._entity_count, self._embedding_dim * 4
                    ).copy()
                    self._quantize_dtype = "fp32"

            self._weight_section_offset = f.tell()

            if self._relation_count > 0:
                weight_size = self._relation_count * 4
                weight_bytes = f.read(weight_size)
                self.truth_weights = np.frombuffer(weight_bytes, dtype=np.float32).copy()

            meta_len = struct.unpack('<I', f.read(4))[0]
            meta_bytes = f.read(meta_len)
            self._parse_metadata(json.loads(meta_bytes.decode('utf-8')))

            # Detect RNIX trailer
            self._rnix_offset = f.tell()
            remaining = f.read(4)
            if remaining == RNIX_MAGIC:
                self._mmap_file = open(path, 'rb')
                self._mmap_obj = mmap.mmap(
                    self._mmap_file.fileno(), 0, access=mmap.ACCESS_READ
                )
                self._rnix = _RNIXReader(self._mmap_obj, self._rnix_offset)

        self._build_indexes()

    def save(self, path: Path, quantize: str = "fp32"):
        """Save to .ramish file (supports requantize command)."""
        path = Path(path)
        self._quantize_dtype = quantize

        int8_scale = 0.0
        if self.embeddings is not None:
            if quantize == "fp32":
                quantized_emb = self.embeddings.astype(np.float32)
            elif quantize == "fp16":
                quantized_emb = self.embeddings.astype(np.float16)
            elif quantize == "int8":
                emb = self.embeddings.astype(np.float32)
                int8_scale = float(np.abs(emb).max())
                if int8_scale < 1e-8:
                    int8_scale = 1.0
                quantized_emb = np.clip(
                    np.round(emb / int8_scale * 127.0), -127, 127
                ).astype(np.int8)
            else:
                raise ValueError(f"Unsupported quantize: {quantize}")
        else:
            quantized_emb = None

        with open(path, 'wb') as f:
            header = self._build_header(quantize=quantize, int8_scale=int8_scale)
            f.write(header)

            self._write_entities(f)
            self._write_relations(f)

            if quantized_emb is not None:
                f.write(quantized_emb.tobytes())

            if self.truth_weights is not None:
                f.write(self.truth_weights.tobytes())

            metadata = self._build_metadata()
            metadata['quantization'] = quantize
            meta_bytes = json.dumps(metadata).encode('utf-8')
            f.write(struct.pack('<I', len(meta_bytes)))
            f.write(meta_bytes)

    def _build_header(self, quantize: str = "fp32", int8_scale: float = 0.0) -> bytes:
        """Build 64-byte header."""
        n_entities = len(self.entities)
        n_relations = len(self.relations)
        dim = self.embeddings.shape[1] // 4 if self.embeddings is not None else 0
        n_rel_types = len(self.relation_types)

        header = bytearray(64)
        header[0:6] = MAGIC
        struct.pack_into('<H', header, 6, VERSION)
        quant_code = QUANT_CODE_MAP.get(quantize, QUANT_FP32)
        struct.pack_into('<I', header, 8, quant_code)
        struct.pack_into('<Q', header, 12, n_entities)
        struct.pack_into('<Q', header, 20, n_relations)
        struct.pack_into('<H', header, 28, dim)
        struct.pack_into('<H', header, 30, n_rel_types)
        struct.pack_into('<d', header, 32, int8_scale)

        return bytes(header)

    def _parse_header(self, header: bytes):
        """Parse header and set internal state."""
        if header[0:6] != MAGIC:
            raise ValueError("Invalid .ramish file: bad magic")

        version = struct.unpack_from('<H', header, 6)[0]
        if version > VERSION:
            raise ValueError(f"Unsupported .ramish version: {version}")

        flags = struct.unpack_from('<I', header, 8)[0]
        self._quant_code = flags & 0x03
        self._entity_count = struct.unpack_from('<Q', header, 12)[0]
        self._relation_count = struct.unpack_from('<Q', header, 20)[0]
        self._embedding_dim = struct.unpack_from('<H', header, 28)[0]
        self._rel_type_count = struct.unpack_from('<H', header, 30)[0]
        self._int8_scale = struct.unpack_from('<d', header, 32)[0]

    # v0.2 Step 7: Header validation
    def _validate_header(self, header: bytes, file_size: int):
        """Validate header values against safety limits before allocating."""
        if self._entity_count > MAX_ENTITY_COUNT:
            raise ValueError(
                f"Entity count {self._entity_count:,} exceeds maximum "
                f"({MAX_ENTITY_COUNT:,}). File may be corrupt."
            )
        if self._relation_count > MAX_RELATION_COUNT:
            raise ValueError(
                f"Relation count {self._relation_count:,} exceeds maximum "
                f"({MAX_RELATION_COUNT:,}). File may be corrupt."
            )
        # Sanity: embedding section alone shouldn't exceed file size
        if self._entity_count > 0 and self._embedding_dim > 0:
            min_embed_bytes = self._entity_count * self._embedding_dim * 4  # fp32 minimum
            if min_embed_bytes > file_size * 2:  # 2x slack for overhead
                raise ValueError(
                    f"Claimed embedding size ({min_embed_bytes:,} bytes) is implausible "
                    f"for file of {file_size:,} bytes. File may be corrupt."
                )

    def _write_entities(self, f):
        """Write entity section."""
        type_to_id = {t: i for i, t in enumerate(self.entity_types)}
        for e in self.entities:
            type_id = type_to_id.get(e.entity_type, 0)
            name_bytes = e.name.encode('utf-8')
            f.write(struct.pack('<H', type_id))
            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)

    def _read_entities(self, f):
        """Read entity section."""
        self.entities = []
        for i in range(self._entity_count):
            type_id = struct.unpack('<H', f.read(2))[0]
            name_len = struct.unpack('<H', f.read(2))[0]
            # v0.2 Step 7: Name length cap
            if name_len > MAX_NAME_LENGTH:
                raise ValueError(
                    f"Entity {i} name length {name_len} exceeds maximum "
                    f"({MAX_NAME_LENGTH}). File may be corrupt."
                )
            name = f.read(name_len).decode('utf-8')
            self.entities.append(Entity(
                id=i,
                external_id=str(i),
                name=name,
                entity_type=str(type_id)
            ))

    def _write_relations(self, f):
        """Write relation section."""
        for r in self.relations:
            f.write(struct.pack('<I', r.head_id))
            f.write(struct.pack('<H', r.relation_type))
            f.write(struct.pack('<I', r.tail_id))

    def _read_relations(self, f):
        """Read relation section."""
        self.relations = []
        for i in range(self._relation_count):
            head_id = struct.unpack('<I', f.read(4))[0]
            rel_type = struct.unpack('<H', f.read(2))[0]
            tail_id = struct.unpack('<I', f.read(4))[0]
            self.relations.append(Relation(
                head_id=head_id,
                relation_type=rel_type,
                tail_id=tail_id
            ))

    def _build_metadata(self) -> dict:
        """Build metadata dictionary."""
        return {
            'entity_types': self.entity_types,
            'relation_types': self.relation_types,
            'relation_type_names': self.relation_type_names
        }

    def _parse_metadata(self, metadata: dict):
        """Parse metadata from loaded file."""
        self.entity_types = metadata.get('entity_types', [])
        self.relation_types = metadata.get('relation_types', {})
        self.relation_type_names = {int(k): v for k, v in metadata.get('relation_type_names', {}).items()}

        for e in self.entities:
            try:
                type_idx = int(e.entity_type)
                if type_idx < len(self.entity_types):
                    e.entity_type = self.entity_types[type_idx]
            except ValueError:
                pass

    def _build_indexes(self):
        """Build internal indexes for fast lookup.

        v0.2: Also builds relation-type indexes, compound lookups,
        and precomputed embedding norms.
        """
        # Existing name/id indexes
        self.name_to_id = {}
        self.name_to_ids = {}
        self.id_to_entity = {}
        for e in self.entities:
            self.name_to_id[e.name.lower()] = e.id
            key = e.name.lower()
            if key not in self.name_to_ids:
                self.name_to_ids[key] = []
            self.name_to_ids[key].append(e.id)
            self.id_to_entity[e.id] = e

        # Existing adjacency list
        self.neighbors = {}
        for i, r in enumerate(self.relations):
            weight = float(self.truth_weights[i]) if self.truth_weights is not None else 0.5

            if r.head_id not in self.neighbors:
                self.neighbors[r.head_id] = []
            self.neighbors[r.head_id].append((r.tail_id, r.relation_type, weight))

            if r.tail_id not in self.neighbors:
                self.neighbors[r.tail_id] = []
            self.neighbors[r.tail_id].append((r.head_id, r.relation_type, weight))

        self._compute_adaptive_threshold()

        # v0.2 Step 3: Relation-type indexes (single pass)
        self.edges_by_rel_type = {}
        self.out_edges = {}
        self.in_edges = {}
        self.targets_by_head_rel = {}
        self.heads_by_tail_rel = {}

        for i, r in enumerate(self.relations):
            # edges_by_rel_type
            if r.relation_type not in self.edges_by_rel_type:
                self.edges_by_rel_type[r.relation_type] = []
            self.edges_by_rel_type[r.relation_type].append(i)

            # out_edges: head_id -> {rel_type -> [indices]}
            if r.head_id not in self.out_edges:
                self.out_edges[r.head_id] = {}
            if r.relation_type not in self.out_edges[r.head_id]:
                self.out_edges[r.head_id][r.relation_type] = []
            self.out_edges[r.head_id][r.relation_type].append(i)

            # in_edges: tail_id -> {rel_type -> [indices]}
            if r.tail_id not in self.in_edges:
                self.in_edges[r.tail_id] = {}
            if r.relation_type not in self.in_edges[r.tail_id]:
                self.in_edges[r.tail_id][r.relation_type] = []
            self.in_edges[r.tail_id][r.relation_type].append(i)

            # Compound lookups
            hr_key = (r.head_id, r.relation_type)
            if hr_key not in self.targets_by_head_rel:
                self.targets_by_head_rel[hr_key] = []
            self.targets_by_head_rel[hr_key].append(r.tail_id)

            tr_key = (r.tail_id, r.relation_type)
            if tr_key not in self.heads_by_tail_rel:
                self.heads_by_tail_rel[tr_key] = []
            self.heads_by_tail_rel[tr_key].append(r.head_id)

        # v0.2 Step 4: Precomputed embedding norms
        if self.embeddings is not None:
            self._embedding_norms = np.linalg.norm(self.embeddings, axis=1)

    def _compute_adaptive_threshold(self):
        """Compute density-adaptive threshold for thick cables."""
        if self.truth_weights is None or len(self.truth_weights) == 0:
            self._avg_weight = 0.5
            self._weight_threshold = 0.5
            return

        self._avg_weight = float(np.mean(self.truth_weights))
        self._weight_threshold = float(np.percentile(self.truth_weights, 75))
        self._weight_threshold = max(self._weight_threshold, 0.1)

    def _build_indexes_geometric(self):
        """Build minimal indexes for geometric mode — entities only, no relation dicts."""
        self.name_to_id = {}
        self.name_to_ids = {}
        self.id_to_entity = {}
        for e in self.entities:
            key = e.name.lower()
            self.name_to_id[key] = e.id
            if key not in self.name_to_ids:
                self.name_to_ids[key] = []
            self.name_to_ids[key].append(e.id)
            self.id_to_entity[e.id] = e

        # Adaptive threshold from mmap'd weights
        if self._weights_mmap is not None and len(self._weights_mmap) > 0:
            # Sample for threshold (don't load all 2B weights into RAM)
            sample_size = min(100_000, len(self._weights_mmap))
            sample_idx = np.linspace(0, len(self._weights_mmap) - 1,
                                     sample_size, dtype=int)
            sample = self._weights_mmap[sample_idx]
            self._avg_weight = float(np.mean(sample))
            self._weight_threshold = float(np.percentile(sample, 75))
            self._weight_threshold = max(self._weight_threshold, 0.1)
        else:
            self._avg_weight = 0.5
            self._weight_threshold = 0.5

    def _binary_search_outgoing(self, head_id: int) -> List[Tuple[int, int, float]]:
        """Find outgoing relations (entity is head) via binary search.

        Returns list of (tail_id, rel_type, weight).
        Only works when relations are sorted by head_id.
        """
        if self._relations_mmap is None or not self._relations_sorted:
            return []

        heads = self._relations_mmap['head_id']
        left = int(np.searchsorted(heads, head_id, side='left'))
        right = int(np.searchsorted(heads, head_id, side='right'))

        results = []
        for i in range(left, right):
            rec = self._relations_mmap[i]
            tail_id = int(rec['tail_id'])
            rel_type = int(rec['rel_type'])
            weight = float(self._weights_mmap[i]) if self._weights_mmap is not None else 0.5
            results.append((tail_id, rel_type, weight))
        return results

    def _binary_search_incoming(self, tail_id: int) -> List[Tuple[int, int, float]]:
        """Find incoming relations (entity is tail) via sorted tail index.

        Returns list of (head_id, rel_type, weight) — note: returns
        the OTHER entity (head_id) as the neighbor, matching the
        convention in self.neighbors.
        """
        if (self._relations_mmap is None or self._tail_sort_idx is None):
            return []

        # _tail_sort_idx maps sorted position → original index
        # The tail_id column reordered by _tail_sort_idx is sorted
        tails_sorted = self._relations_mmap['tail_id'][self._tail_sort_idx]
        left = int(np.searchsorted(tails_sorted, tail_id, side='left'))
        right = int(np.searchsorted(tails_sorted, tail_id, side='right'))

        results = []
        for pos in range(left, right):
            orig_idx = int(self._tail_sort_idx[pos])
            rec = self._relations_mmap[orig_idx]
            head_id = int(rec['head_id'])
            rel_type = int(rec['rel_type'])
            weight = float(self._weights_mmap[orig_idx]) if self._weights_mmap is not None else 0.5
            results.append((head_id, rel_type, weight))
        return results

    def _binary_search_relations(self, entity_id: int) -> List[Tuple[int, int, float]]:
        """Find ALL relations for an entity — both directions.

        Returns list of (neighbor_id, rel_type, weight), matching
        the format of self.neighbors[entity_id] in full-load mode.
        Merges outgoing (entity is head) + incoming (entity is tail).
        """
        outgoing = self._binary_search_outgoing(entity_id)
        incoming = self._binary_search_incoming(entity_id)
        return outgoing + incoming

    def _binary_search_edge(self, head_id: int, tail_id: int,
                            rel_type: Optional[int] = None) -> Optional[Tuple[float, int]]:
        """Check if a specific edge exists via binary search.

        Searches both directions: head→tail and tail→head.
        Returns (weight, rel_type) if found, None otherwise.
        """
        if self._relations_mmap is None or not self._relations_sorted:
            return None

        # Forward: head_id as head
        heads = self._relations_mmap['head_id']
        left = int(np.searchsorted(heads, head_id, side='left'))
        right = int(np.searchsorted(heads, head_id, side='right'))

        for i in range(left, right):
            rec = self._relations_mmap[i]
            if int(rec['tail_id']) == tail_id:
                rt = int(rec['rel_type'])
                if rel_type is None or rt == rel_type:
                    w = float(self._weights_mmap[i]) if self._weights_mmap is not None else 0.5
                    return (w, rt)

        # Reverse: head_id as tail (the edge was stored as tail_id→head_id)
        left = int(np.searchsorted(heads, tail_id, side='left'))
        right = int(np.searchsorted(heads, tail_id, side='right'))

        for i in range(left, right):
            rec = self._relations_mmap[i]
            if int(rec['tail_id']) == head_id:
                rt = int(rec['rel_type'])
                if rel_type is None or rt == rel_type:
                    w = float(self._weights_mmap[i]) if self._weights_mmap is not None else 0.5
                    return (w, rt)

        return None

    @property
    def is_geometric_mode(self) -> bool:
        """Whether this file was loaded in geometric (mmap) mode."""
        return self._geometric_mode

    @property
    def has_rnix(self) -> bool:
        """Whether RNIX name index is available."""
        return self._rnix is not None

    def close(self):
        """Release mmap resources. Call when done with geometric mode files.

        On Windows, numpy memmaps hold file handles independently of
        mmap.mmap. Explicit del + gc.collect ensures file locks release.
        """
        if self._mmap_obj is not None:
            self._mmap_obj.close()
            self._mmap_obj = None
        if self._mmap_file is not None:
            self._mmap_file.close()
            self._mmap_file = None
        self._rnix = None
        # Explicitly delete memmap arrays — Windows holds file locks
        # until the numpy memmap object is garbage collected
        if self._relations_mmap is not None:
            del self._relations_mmap
            self._relations_mmap = None
        if self._weights_mmap is not None:
            del self._weights_mmap
            self._weights_mmap = None
        if self._tail_sort_idx is not None:
            del self._tail_sort_idx
            self._tail_sort_idx = None
        # Embeddings may be memmap'd in geometric mode
        if self._geometric_mode and self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
            if self._embedding_norms is not None:
                del self._embedding_norms
                self._embedding_norms = None
        import gc
        gc.collect()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Fix 3: Centralized name resolution ─────────────────────────

    def resolve_name(self, name: str) -> Tuple[List[int], bool]:
        """Resolve a name to entity IDs using the multi-map.

        Returns (entity_ids, ambiguous) where ambiguous=True when
        multiple entities share the same name.  Empty list if no match.
        Uses RNIX binary search when available (geometric mode or
        standard load with RNIX trailer detected).
        """
        # RNIX fast path — zero-RAM binary search on sorted name index
        if self._rnix is not None:
            ids = self._rnix.resolve_ids(name)
            if ids:
                return (ids, len(ids) > 1)
            # Fall through to dict for RNIX miss (shouldn't happen, but safe)

        key = name.lower()
        ids = self.name_to_ids.get(key, [])
        if ids:
            return (list(ids), len(ids) > 1)
        return ([], False)

    def resolve_name_fuzzy(self, name: str, max_suggestions: int = 10) -> Tuple[List[int], bool, List[str]]:
        """Resolve with fuzzy fallback.

        Returns (entity_ids, ambiguous, suggestions).
        If exact match found: suggestions is empty.
        If no exact match: entity_ids is empty, suggestions has close matches.
        """
        ids, ambiguous = self.resolve_name(name)
        if ids:
            return (ids, ambiguous, [])

        # Fuzzy: RNIX prefix search or dict substring match
        if self._rnix is not None:
            results = self._rnix.search(name, max_results=max_suggestions)
            suggestions = [r[2] for r in results]  # (eid, etype, name, match_type)
            return ([], False, suggestions)

        key = name.lower()
        suggestions = [n for n in self.name_to_ids.keys() if key in n]
        return ([], False, suggestions[:max_suggestions])

    # ── Progressive Name Index (PNI) ──────────────────────────────

    def load_pni(self, path: str):
        """Load a Progressive Name Index from an external file.

        Enables progressive narrowing and fast exact name resolution.
        Used for large files where dict-based name lookup is impractical.
        For small files, the built-in dict indexes work fine without PNI.
        """
        from processing.ramish.progressive_index import PNIReader
        self._pni = PNIReader(path)

    def build_pni(self):
        """Build a PNI from current entities (in-memory, for testing).

        For production, PNI is written as a trailer section during
        .ramish assembly and loaded via load_pni() or detected on load.
        """
        from processing.ramish.progressive_index import PNIWriter, PNIReader
        import tempfile
        writer = PNIWriter()
        for e in self.entities:
            writer.add(e.name, e.id)
        tmp = tempfile.NamedTemporaryFile(suffix='.pni', delete=False)
        writer.write(tmp.name)
        self._pni = PNIReader(tmp.name)

    @property
    def has_pni(self) -> bool:
        """Whether a Progressive Name Index is loaded."""
        return self._pni is not None

    def narrow(self, text: str, max_results: int = 20) -> List[Tuple[int, str, str, int]]:
        """Progressive name narrowing. Returns candidates sorted by hub weight.

        Each result: (entity_id, display_name, entity_type, connection_count)

        Uses RNIX if available, then PNI, falls back to dict-based prefix search.
        """
        if self._rnix is not None:
            rnix_results = self._rnix.search(text, max_results=max_results * 2)
            entity_ids = [eid for eid, _, _, _ in rnix_results]
        elif self._pni is not None:
            entity_ids = self._pni.narrow(text, max_results=max_results * 2)
        else:
            # Dict fallback: prefix match on name_to_ids
            text_lower = text.lower().strip()
            entity_ids = []
            for name, ids in self.name_to_ids.items():
                if name.startswith(text_lower):
                    entity_ids.extend(ids)
                if len(entity_ids) >= max_results * 2:
                    break

        # Resolve to display info + sort by hub weight
        candidates = []
        for eid in entity_ids:
            entity = self.id_to_entity.get(eid)
            if entity:
                if self._geometric_mode and self._relations_sorted:
                    # Binary search for connection count
                    conn_count = len(self._binary_search_relations(eid))
                else:
                    conn_count = len(self.neighbors.get(eid, []))
                candidates.append((eid, entity.name, entity.entity_type, conn_count))

        candidates.sort(key=lambda x: -x[3])
        return candidates[:max_results]

    def autocomplete(self, prefix: str, max_results: int = 10) -> List[Tuple[int, str, str, int]]:
        """Autocomplete for interactive use. Alias for narrow()."""
        return self.narrow(prefix, max_results=max_results)

    # ── Step 8: Geometric query engine ────────────────────────────

    def query(self, query_text: str, topk: int = 10) -> List[QueryResult]:
        """Query the knowledge graph — hybrid geometric + graph rerank.

        v0.2: Uses geometric cosine retrieval seeded by lexical match,
        then injects known graph neighbors and reranks.
        Falls back to wave propagation if no embeddings.
        """
        query_lower = query_text.lower()

        # === Three-tier lexical seed matching ===
        #
        # Tier 1: Entity name boundary-matches inside query text
        #   "What about Led Zeppelin?" → seeds "led zeppelin"
        #
        # Tier 2: Query boundary-matches inside entity names (bidirectional)
        #   query "session_briefing" → seeds "session_briefing.py"
        #   query "config_reader" → seeds "config_reader.py", "config_reader.js"
        #
        # Tier 3: Individual query words boundary-match inside entity names
        #   query "session briefing analysis" → seeds entities containing "session", "briefing"
        #
        # All tiers use \b word boundaries. Underscore is a word character,
        # so "get" won't match inside "budget_getter" via tier 1, and
        # "ramish_explorer" won't match inside "test_ramish_explorer" via tier 2.
        # The . in "file.py" IS a boundary, so "file" matches "file.py".

        seed_entities = []

        # Tier 0: RNIX binary search (geometric mode or standard load with RNIX)
        # Zero-RAM sorted name lookup — skips regex scanning entirely
        if self._rnix is not None:
            rnix_ids = self._rnix.resolve_ids(query_lower)
            if rnix_ids:
                seed_entities = rnix_ids
            else:
                # Try prefix match for partial query terms
                rnix_results = self._rnix.search(query_lower, max_results=10)
                seed_entities = [eid for eid, _, _, _ in rnix_results]

        # Tier 0b: PNI resolution (if RNIX not available)
        if not seed_entities and self._pni is not None:
            pni_ids = self._pni.narrow(query_lower)
            if pni_ids:
                seed_entities = pni_ids

        # In geometric mode, skip the regex tiers — name_to_ids may be
        # populated from entity load but regex scanning 385M names is not viable.
        # RNIX/PNI handles seed resolution; if no seeds found, return empty.
        if self._geometric_mode and not seed_entities:
            # Last resort: try individual query words through RNIX
            if self._rnix is not None:
                words = [w for w in query_lower.split() if len(w) > 3]
                for w in words:
                    word_ids = self._rnix.resolve_ids(w)
                    seed_entities.extend(word_ids)
                    if not word_ids:
                        word_results = self._rnix.search(w, max_results=5)
                        seed_entities.extend(eid for eid, _, _, _ in word_results)
                    if len(seed_entities) >= 5:
                        break
            if not seed_entities:
                return []

        # Tier 1: entity name ⊂ query (skip if PNI already found seeds)
        if not seed_entities:
            for name, ids in self.name_to_ids.items():
                if re.search(r'\b' + re.escape(name) + r'\b', query_lower):
                    seed_entities.extend(ids)

        # Tier 2: query ⊂ entity name (only if tier 1 found nothing)
        if not seed_entities and len(query_lower) > 2:
            query_pattern = re.compile(r'\b' + re.escape(query_lower) + r'\b')
            for name, ids in self.name_to_ids.items():
                if query_pattern.search(name):
                    seed_entities.extend(ids)

        # Tier 3: query words ⊂ entity names (only if tiers 1-2 found nothing)
        if not seed_entities:
            words = [w for w in query_lower.split() if len(w) > 3]
            for name, ids in self.name_to_ids.items():
                for w in words:
                    if re.search(r'\b' + re.escape(w) + r'\b', name):
                        seed_entities.extend(ids)
                        break
                if len(seed_entities) >= 5:
                    break

        if not seed_entities:
            return []

        # Deduplicate seeds while preserving order
        seen = set()
        unique_seeds = []
        for sid in seed_entities:
            if sid not in seen:
                seen.add(sid)
                unique_seeds.append(sid)
        seed_entities = unique_seeds

        # v0.2: If we have embeddings, use geometric retrieval
        if self.embeddings is not None and self._embedding_norms is not None:
            return self._geometric_query(seed_entities, topk)

        # Fallback: wave propagation (v0.1 behavior)
        return self._wave_query(seed_entities, topk)

    def _geometric_query(self, seed_ids: List[int], topk: int) -> List[QueryResult]:
        """Hybrid geometric query: cosine retrieval + graph neighbor injection + rerank.

        In geometric mode, uses binary search on mmap'd sorted relations
        instead of the in-memory neighbors dict.
        """
        results = []
        seen = set()

        for seed_id in seed_ids:
            seed_entity = self.id_to_entity.get(seed_id)
            if not seed_entity:
                continue

            seed_emb = self.embeddings[seed_id]
            seed_norm = self._embedding_norms[seed_id]
            if seed_norm < 1e-8:
                continue

            # Cosine similarity to all entities
            # In geometric mode with large files, embeddings is a memmap —
            # the OS pages in only the rows touched by the matmul.
            emb_f32 = self.embeddings
            if emb_f32.dtype != np.float32:
                # Dequantize seed for dot product
                seed_emb = seed_emb.astype(np.float32)
                emb_f32 = self.embeddings.astype(np.float32)

            similarities = emb_f32 @ seed_emb / (self._embedding_norms * float(seed_norm) + 1e-8)
            similarities[seed_id] = -1.0  # exclude self

            # Get top geometric candidates
            geo_top = np.argsort(similarities)[::-1][:topk * 3]

            # Build candidate pool: geometric candidates + known graph neighbors
            candidate_pool = {}  # entity_id -> (score, rel_name, is_graph_edge)

            # Inject known graph neighbors first (★ = known edge)
            # Geometric mode: binary search on sorted mmap'd relations
            # Full-load mode: use in-memory neighbors dict
            if self._geometric_mode and self._relations_sorted:
                neighbors = self._binary_search_relations(seed_id)
            elif seed_id in self.neighbors:
                neighbors = self.neighbors[seed_id]
            else:
                neighbors = []

            for neighbor_id, rel_type, weight in neighbors:
                rel_name = self.relation_type_names.get(rel_type, f"relation_{rel_type}")
                # Score: use truth weight boosted by geometric similarity
                geo_sim = float(similarities[neighbor_id]) if neighbor_id < len(similarities) else 0.0
                combined_score = 0.6 * weight + 0.4 * max(0, geo_sim)
                key = (seed_id, rel_type, neighbor_id)
                if key not in seen:
                    candidate_pool[neighbor_id] = (combined_score, rel_name, True)
                    seen.add(key)

            # Add geometric candidates (~ = inferred by geometry)
            for idx in geo_top:
                idx_int = int(idx)
                if idx_int == seed_id or idx_int in candidate_pool:
                    continue
                neighbor_entity = self.id_to_entity.get(idx_int)
                if not neighbor_entity:
                    continue

                sim = float(similarities[idx_int])
                if sim < 0.01:
                    break

                # Infer likely relation type from entity type clustering
                rel_name = self._infer_relation(seed_entity, neighbor_entity)
                candidate_pool[idx_int] = (sim * 0.5, f"~{rel_name}", False)

            # Sort by score and build results
            sorted_candidates = sorted(candidate_pool.items(), key=lambda x: -x[1][0])

            for entity_id, (score, rel_name, is_edge) in sorted_candidates[:topk]:
                neighbor_entity = self.id_to_entity.get(entity_id)
                if not neighbor_entity:
                    continue
                supporting_paths = max(1, int(score * 1000))
                results.append(QueryResult(
                    subject=seed_entity.name,
                    relation=rel_name,
                    object=neighbor_entity.name,
                    truth_weight=min(1.0, score),
                    supporting_paths=supporting_paths
                ))

        results.sort(key=lambda r: -r.truth_weight)
        return results[:topk]

    def _infer_relation(self, source: Any, target: Any) -> str:
        """Infer most likely relation name between two entities based on type pair.

        In geometric mode, samples binary search results instead of out_edges.
        """
        src_type = source.entity_type
        tgt_type = target.entity_type

        if self._geometric_mode and self._relations_sorted:
            # Sample source's outgoing relations to find common rel_type to target's type
            rels = self._binary_search_outgoing(source.id)
            for tail_id, rel_type, _ in rels[:20]:  # sample first 20
                tail_entity = self.id_to_entity.get(tail_id)
                if tail_entity and tail_entity.entity_type == tgt_type:
                    return self.relation_type_names.get(rel_type, f"relation_{rel_type}")
        elif source.id in self.out_edges:
            # Full-load mode: use prebuilt out_edges index
            for rel_type, indices in self.out_edges[source.id].items():
                for idx in indices[:5]:  # sample
                    r = self.relations[idx]
                    tail_entity = self.id_to_entity.get(r.tail_id)
                    if tail_entity and tail_entity.entity_type == tgt_type:
                        return self.relation_type_names.get(rel_type, f"relation_{rel_type}")

        # Fallback: just use "similar"
        return "similar"

    def _get_neighbors(self, entity_id: int) -> List[Tuple[int, int, float]]:
        """Get neighbors for an entity — abstracts geometric vs full-load mode.

        Returns list of (neighbor_id, rel_type, weight).
        """
        if self._geometric_mode and self._relations_sorted:
            return self._binary_search_relations(entity_id)
        return self.neighbors.get(entity_id, [])

    def _wave_query(self, seed_ids: List[int], topk: int) -> List[QueryResult]:
        """Perform wave propagation query (v0.1 fallback).

        Uses _get_neighbors() which transparently handles both
        geometric mode (binary search) and full-load mode (dict).
        """
        activation = {eid: 1.0 for eid in seed_ids}

        for hop in range(2):
            new_activation = {}
            for entity_id, act in activation.items():
                neighbors = self._get_neighbors(entity_id)
                for neighbor_id, rel_type, weight in neighbors:
                    prop = act * weight * 0.5
                    if neighbor_id in new_activation:
                        new_activation[neighbor_id] = max(new_activation[neighbor_id], prop)
                    else:
                        new_activation[neighbor_id] = prop

            for eid, act in new_activation.items():
                if eid not in activation or new_activation[eid] > activation[eid]:
                    activation[eid] = new_activation[eid]

        results = []
        seen = set()

        for seed_id in seed_ids:
            seed_entity = self.id_to_entity.get(seed_id)
            if not seed_entity:
                continue

            neighbors = self._get_neighbors(seed_id)
            if not neighbors:
                continue

            for neighbor_id, rel_type, weight in neighbors:
                key = (seed_id, rel_type, neighbor_id)
                if key in seen:
                    continue
                seen.add(key)

                neighbor_entity = self.id_to_entity.get(neighbor_id)
                if not neighbor_entity:
                    continue

                rel_name = self.relation_type_names.get(rel_type, f"relation_{rel_type}")
                supporting_paths = max(1, int(weight * 1000))

                results.append(QueryResult(
                    subject=seed_entity.name,
                    relation=rel_name,
                    object=neighbor_entity.name,
                    truth_weight=weight,
                    supporting_paths=supporting_paths
                ))

        results.sort(key=lambda r: -r.truth_weight)
        return results[:topk]

    def validate_claim(self, subject: str, relation: str, object: str) -> ValidationResult:
        """Validate a specific claim.

        In geometric mode, uses RNIX for name resolution and binary search
        on sorted mmap'd relations for edge lookup.
        """
        # Name resolution — resolve_name already uses RNIX when available
        subj_ids, _ = self.resolve_name(subject)
        obj_ids, _ = self.resolve_name(object)

        if not subj_ids or not obj_ids:
            return ValidationResult(
                truth_weight=0.0, supporting_paths=0,
                verdict="UNKNOWN - entities not found"
            )

        rel_type = self.relation_types.get(relation)
        if rel_type is None:
            for name, rt in self.relation_types.items():
                if relation.lower() in name.lower() or name.lower() in relation.lower():
                    rel_type = rt
                    break

        weight = 0.0
        found = False

        for subj_id in subj_ids:
            for obj_id in obj_ids:
                if self._geometric_mode and self._relations_sorted:
                    # Binary search on sorted mmap'd relations
                    result = self._binary_search_edge(subj_id, obj_id, rel_type)
                    if result is not None:
                        w, rt = result
                        weight = max(weight, w)
                        found = True
                elif subj_id in self.neighbors:
                    # Full-load mode: scan in-memory neighbors
                    for neighbor_id, rt, w in self.neighbors[subj_id]:
                        if neighbor_id == obj_id:
                            if rel_type is None or rt == rel_type:
                                weight = max(weight, w)
                                found = True

        if not found:
            return ValidationResult(
                truth_weight=0.0, supporting_paths=0,
                verdict="NOT FOUND - no such relation exists"
            )

        supporting_paths = max(1, int(weight * 1000))
        verdict = get_verdict(weight, supporting_paths)

        return ValidationResult(
            truth_weight=weight,
            supporting_paths=supporting_paths,
            verdict=verdict
        )

    def get_stats(self) -> RamishStats:
        """Get statistics about this file."""
        n_entities = len(self.entities)
        # In geometric mode, relations aren't loaded into self.relations
        n_relations = self._relation_count if self._geometric_mode else len(self.relations)
        dim = self.embeddings.shape[1] // 4 if self.embeddings is not None else 0

        file_size = 0.0
        if self._loaded_path and self._loaded_path.exists():
            file_size = self._loaded_path.stat().st_size / (1024 * 1024)

        # Weight distribution — use mmap'd weights in geometric mode
        weights = None
        if self._geometric_mode and self._weights_mmap is not None and len(self._weights_mmap) > 0:
            weights = self._weights_mmap
        elif self.truth_weights is not None and len(self.truth_weights) > 0:
            weights = self.truth_weights

        if weights is not None and len(weights) > 0:
            # For large mmap'd arrays, sample to avoid loading everything
            if len(weights) > 1_000_000:
                sample_idx = np.linspace(0, len(weights) - 1, 100_000, dtype=int)
                sample = weights[sample_idx]
            else:
                sample = weights
            high = 100 * float((sample > 0.7).sum()) / len(sample)
            medium = 100 * float(((sample > 0.3) & (sample <= 0.7)).sum()) / len(sample)
            low = 100 * float((sample <= 0.3).sum()) / len(sample)
        else:
            high = medium = low = 0.0

        return RamishStats(
            entity_count=n_entities,
            relation_count=n_relations,
            embedding_dim=dim,
            file_size_mb=file_size,
            compression_ratio=1.0,
            high_confidence_pct=high,
            medium_confidence_pct=medium,
            low_confidence_pct=low
        )

    def audit(self) -> AuditResult:
        """Run a data quality audit.

        In geometric mode, uses sampled mmap analysis instead of
        iterating over all relations in memory.
        """
        issues = []
        recommendations = []

        if self._geometric_mode:
            # Sampled audit for large mmap'd files
            n_entities = len(self.entities)
            n_relations = self._relation_count

            # Orphan detection via sampled relations
            if self._relations_mmap is not None and n_relations > 0:
                sample_size = min(500_000, n_relations)
                if sample_size < n_relations:
                    sample_idx = np.linspace(0, n_relations - 1, sample_size, dtype=int)
                    heads = self._relations_mmap['head_id'][sample_idx]
                    tails = self._relations_mmap['tail_id'][sample_idx]
                else:
                    heads = self._relations_mmap['head_id']
                    tails = self._relations_mmap['tail_id']
                connected = set(int(h) for h in heads) | set(int(t) for t in tails)
                # Estimate orphans from sample
                coverage = len(connected) / max(n_entities, 1)
                est_orphans = max(0, n_entities - int(coverage * n_entities))
                if est_orphans > 0:
                    pct = 100 * est_orphans / max(n_entities, 1)
                    severity = "high" if pct > 20 else "medium" if pct > 5 else "low"
                    sampled_note = " (sampled estimate)" if sample_size < n_relations else ""
                    issues.append(AuditIssue(
                        severity=severity,
                        description=f"~{est_orphans} entities ({pct:.1f}%) have no connections{sampled_note}",
                        affected_count=est_orphans
                    ))
                    if pct > 10:
                        recommendations.append("Consider filtering orphan entities or checking data extraction")

            # Weight analysis via sampled mmap
            weights = self._weights_mmap
            if weights is not None and len(weights) > 0:
                sample_size = min(100_000, len(weights))
                if sample_size < len(weights):
                    sample_idx = np.linspace(0, len(weights) - 1, sample_size, dtype=int)
                    sample = weights[sample_idx]
                else:
                    sample = weights
                threshold = self._weight_threshold or 0.5
                low_threshold = threshold * 0.3
                low_weight_pct = 100 * float((sample < low_threshold).sum()) / len(sample)
                if low_weight_pct > 0:
                    est_low = int(low_weight_pct * n_relations / 100)
                    severity = "medium" if low_weight_pct > 30 else "low"
                    issues.append(AuditIssue(
                        severity=severity,
                        description=f"~{est_low} relations ({low_weight_pct:.1f}%) have very low confidence (sampled)",
                        affected_count=est_low
                    ))
                    if low_weight_pct > 20:
                        recommendations.append("Review low-confidence relations for potential data quality issues")

            # Degree imbalance: sample entity degrees via binary search
            if self._relations_sorted and self._relations_mmap is not None:
                sample_eids = np.random.choice(n_entities, min(1000, n_entities), replace=False)
                degrees = []
                for eid in sample_eids:
                    rels = self._binary_search_relations(int(eid))
                    degrees.append(len(rels))
                if degrees:
                    max_deg = max(degrees)
                    degrees_sorted = sorted(degrees)
                    median_deg = degrees_sorted[len(degrees_sorted) // 2]
                    if max_deg > max(1, median_deg) * 100:
                        issues.append(AuditIssue(
                            severity="low",
                            description=f"Extreme hub imbalance: sampled max degree {max_deg} vs median {median_deg}",
                            affected_count=1
                        ))

        else:
            # Full-load mode: original audit path
            connected = set()
            for r in self.relations:
                connected.add(r.head_id)
                connected.add(r.tail_id)

            orphans = len(self.entities) - len(connected)
            if orphans > 0:
                pct = 100 * orphans / max(len(self.entities), 1)
                severity = "high" if pct > 20 else "medium" if pct > 5 else "low"
                issues.append(AuditIssue(
                    severity=severity,
                    description=f"{orphans} entities ({pct:.1f}%) have no connections",
                    affected_count=orphans
                ))
                if pct > 10:
                    recommendations.append("Consider filtering orphan entities or checking data extraction")

            if self.truth_weights is not None:
                threshold = self._weight_threshold or 0.5
                low_threshold = threshold * 0.3
                low_weight = (self.truth_weights < low_threshold).sum()
                if low_weight > 0:
                    pct = 100 * low_weight / len(self.truth_weights)
                    severity = "medium" if pct > 30 else "low"
                    issues.append(AuditIssue(
                        severity=severity,
                        description=f"{low_weight} relations ({pct:.1f}%) have very low confidence",
                        affected_count=int(low_weight)
                    ))
                    if pct > 20:
                        recommendations.append("Review low-confidence relations for potential data quality issues")

            degree_counts = {}
            for eid in self.neighbors:
                degree_counts[eid] = len(self.neighbors[eid])

            if degree_counts:
                max_degree = max(degree_counts.values())
                median_degree = sorted(degree_counts.values())[len(degree_counts) // 2]
                if max_degree > median_degree * 100:
                    issues.append(AuditIssue(
                        severity="low",
                        description=f"Extreme hub imbalance: max degree {max_degree} vs median {median_degree}",
                        affected_count=1
                    ))

        if not issues:
            overall_score = 1.0
        else:
            severity_weights = {"high": 0.3, "medium": 0.1, "low": 0.05}
            penalty = sum(severity_weights.get(i.severity, 0) for i in issues)
            overall_score = max(0, 1.0 - penalty)

        return AuditResult(
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )

    def get_top_hubs(self, n: int = 10) -> List[HubInfo]:
        """Get the top hub entities by degree.

        In geometric mode with sorted relations, uses numpy unique counts
        on the mmap'd head_id column for efficient degree computation.
        """
        threshold = self._weight_threshold or 0.5

        if self._geometric_mode and self._relations_mmap is not None and self._relations_sorted:
            # Count degree (both directions) using head_id + tail_id columns
            heads = self._relations_mmap['head_id']
            tails = self._relations_mmap['tail_id']
            n_rels = len(heads)

            if n_rels > 10_000_000:
                sample_size = min(5_000_000, n_rels)
                sample_idx = np.linspace(0, n_rels - 1, sample_size, dtype=int)
                all_ids = np.concatenate([heads[sample_idx], tails[sample_idx]])
            else:
                all_ids = np.concatenate([np.array(heads), np.array(tails)])

            unique_ids, counts = np.unique(all_ids, return_counts=True)

            # Sort by count descending, take top n
            top_idx = np.argsort(counts)[::-1][:n * 2]  # extra for fallback

            hubs = []
            for idx in top_idx:
                eid = int(unique_ids[idx])
                entity = self.id_to_entity.get(eid)
                if not entity:
                    continue

                # Get actual relations for this hub to compute thick/loose
                rels = self._binary_search_relations(eid)
                degree = len(rels)
                thick = sum(1 for _, _, w in rels if w >= threshold)
                loose = degree - thick
                avg_w = sum(w for _, _, w in rels) / degree if degree > 0 else 0.0

                hubs.append(HubInfo(
                    entity_id=eid,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    degree=degree,
                    thick_cables=thick,
                    loose_threads=loose,
                    avg_weight=avg_w
                ))
                if len(hubs) >= n:
                    break

            return hubs

        # Full-load mode: original path
        degree_counts = {}
        for eid in self.neighbors:
            degree_counts[eid] = len(self.neighbors[eid])

        top_ids = sorted(degree_counts.keys(), key=lambda x: -degree_counts[x])[:n]

        hubs = []
        for eid in top_ids:
            entity = self.id_to_entity.get(eid)
            if not entity:
                continue

            thick = 0
            loose = 0
            weights = []
            for _, _, weight in self.neighbors[eid]:
                weights.append(weight)
                if weight >= threshold:
                    thick += 1
                else:
                    loose += 1

            avg_weight = sum(weights) / len(weights) if weights else 0

            hubs.append(HubInfo(
                entity_id=eid,
                name=entity.name,
                entity_type=entity.entity_type,
                degree=degree_counts[eid],
                thick_cables=thick,
                loose_threads=loose,
                avg_weight=avg_weight
            ))

        return hubs

    def list_entities(self, type_filter: Optional[str] = None, limit: int = 20) -> List[Any]:
        """List entities, optionally filtered by type."""
        result = []
        for e in self.entities:
            if type_filter and e.entity_type.lower() != type_filter.lower():
                continue
            result.append(e)
            if len(result) >= limit:
                break
        return result

    def get_relations(self, entity_name: str, entity_id: Optional[int] = None) -> List[Any]:
        """Get all relations for an entity.

        If entity_id is provided, uses it directly (disambiguation).
        Otherwise resolves by name — if ambiguous, returns relations for ALL
        matching entities with source_entity_id set on each result.

        In geometric mode, uses binary search on sorted mmap'd relations
        instead of the in-memory neighbors dict.
        """
        @dataclass
        class RelInfo:
            relation: str
            target: str
            truth_weight: float
            source_entity_id: Optional[int] = None

        if entity_id is not None:
            resolve_ids = [entity_id]
        else:
            # resolve_name already uses RNIX when available
            resolve_ids, _ = self.resolve_name(entity_name)

        if not resolve_ids:
            return []

        tag_source = len(resolve_ids) > 1
        results = []
        for eid in resolve_ids:
            if self._geometric_mode and self._relations_sorted:
                # Binary search on sorted mmap'd relations
                neighbors = self._binary_search_relations(eid)
                for neighbor_id, rel_type, weight in neighbors:
                    neighbor = self.id_to_entity.get(neighbor_id)
                    rel_name = self.relation_type_names.get(rel_type, f"relation_{rel_type}")
                    results.append(RelInfo(
                        relation=rel_name,
                        target=neighbor.name if neighbor else f"entity_{neighbor_id}",
                        truth_weight=weight,
                        source_entity_id=eid if tag_source else None
                    ))
            elif eid in self.neighbors:
                # Full-load mode: in-memory neighbors dict
                for neighbor_id, rel_type, weight in self.neighbors[eid]:
                    neighbor = self.id_to_entity.get(neighbor_id)
                    rel_name = self.relation_type_names.get(rel_type, f"relation_{rel_type}")
                    results.append(RelInfo(
                        relation=rel_name,
                        target=neighbor.name if neighbor else f"entity_{neighbor_id}",
                        truth_weight=weight,
                        source_entity_id=eid if tag_source else None
                    ))

        return sorted(results, key=lambda r: -r.truth_weight)
