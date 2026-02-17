"""
.ramish File Format Reader

Binary format reader for portable knowledge graphs with truth weights.
This is the read-only version for the ramish-explorer package.
"""
import struct
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .models import Entity, Relation
from .quantize import EmbeddingQuantizer


MAGIC = b'RAMISH'
VERSION = 1

# Quantization type codes stored in header flags field
QUANT_FP32 = 0
QUANT_FP16 = 1
QUANT_INT8 = 2
QUANT_DTYPE_MAP = {QUANT_FP32: "fp32", QUANT_FP16: "fp16", QUANT_INT8: "int8"}
QUANT_CODE_MAP = {"fp32": QUANT_FP32, "fp16": QUANT_FP16, "int8": QUANT_INT8}


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


class RamishFile:
    """Handler for .ramish file format (read-only)."""

    def __init__(self):
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

    @classmethod
    def load(cls, path: str) -> 'RamishFile':
        """Load from .ramish file.

        Handles both legacy (fp32) and quantized (fp16/int8) formats.
        Embeddings are always dequantized to fp32 on load so all
        downstream code works without modification.
        """
        path = Path(path)
        rf = cls()
        rf._loaded_path = path

        with open(path, 'rb') as f:
            header = f.read(64)
            rf._parse_header(header)

            rf._read_entities(f)
            rf._read_relations(f)

            if rf._entity_count > 0 and rf._embedding_dim > 0:
                total_components = rf._entity_count * rf._embedding_dim * 4
                quant_code = rf._quant_code

                if quant_code == QUANT_FP16:
                    embed_bytes = f.read(total_components * 2)
                    raw = np.frombuffer(embed_bytes, dtype=np.float16).reshape(
                        rf._entity_count, rf._embedding_dim * 4
                    )
                    rf.embeddings = EmbeddingQuantizer.dequantize({
                        "embeddings": raw, "dtype": "fp16"
                    })
                    rf._quantize_dtype = "fp16"
                elif quant_code == QUANT_INT8:
                    embed_bytes = f.read(total_components)
                    raw = np.frombuffer(embed_bytes, dtype=np.int8).reshape(
                        rf._entity_count, rf._embedding_dim * 4
                    )
                    rf.embeddings = EmbeddingQuantizer.dequantize({
                        "embeddings": raw, "dtype": "int8", "scale": rf._int8_scale
                    })
                    rf._quantize_dtype = "int8"
                else:
                    embed_bytes = f.read(total_components * 4)
                    rf.embeddings = np.frombuffer(embed_bytes, dtype=np.float32).reshape(
                        rf._entity_count, rf._embedding_dim * 4
                    ).copy()
                    rf._quantize_dtype = "fp32"

            if rf._relation_count > 0:
                weight_size = rf._relation_count * 4
                weight_bytes = f.read(weight_size)
                rf.truth_weights = np.frombuffer(weight_bytes, dtype=np.float32).copy()

            meta_len = struct.unpack('<I', f.read(4))[0]
            meta_bytes = f.read(meta_len)
            rf._parse_metadata(json.loads(meta_bytes.decode('utf-8')))

        rf._build_indexes()

        return rf

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
        """Build internal indexes for fast lookup."""
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

        self.neighbors = {}
        for i, r in enumerate(self.relations):
            weight = self.truth_weights[i] if self.truth_weights is not None else 0.5

            if r.head_id not in self.neighbors:
                self.neighbors[r.head_id] = []
            self.neighbors[r.head_id].append((r.tail_id, r.relation_type, weight))

            if r.tail_id not in self.neighbors:
                self.neighbors[r.tail_id] = []
            self.neighbors[r.tail_id].append((r.head_id, r.relation_type, weight))

        self._compute_adaptive_threshold()

    def _compute_adaptive_threshold(self):
        """Compute density-adaptive threshold for thick cables."""
        if self.truth_weights is None or len(self.truth_weights) == 0:
            self._avg_weight = 0.5
            self._weight_threshold = 0.5
            return

        self._avg_weight = float(np.mean(self.truth_weights))
        self._weight_threshold = float(np.percentile(self.truth_weights, 75))
        self._weight_threshold = max(self._weight_threshold, 0.1)

    def query(self, query_text: str, topk: int = 10) -> List[QueryResult]:
        """Query the knowledge graph with natural language."""
        query_lower = query_text.lower()

        seed_entities = []
        for name, entity_id in self.name_to_id.items():
            if name in query_lower:
                seed_entities.append(entity_id)

        if not seed_entities:
            for name, entity_id in self.name_to_id.items():
                words = query_lower.split()
                if any(w in name for w in words if len(w) > 3):
                    seed_entities.append(entity_id)
                    if len(seed_entities) >= 3:
                        break

        if not seed_entities:
            return []

        return self._wave_query(seed_entities, topk)

    def _wave_query(self, seed_ids: List[int], topk: int) -> List[QueryResult]:
        """Perform wave propagation query."""
        activation = {eid: 1.0 for eid in seed_ids}

        for hop in range(2):
            new_activation = {}
            for entity_id, act in activation.items():
                if entity_id not in self.neighbors:
                    continue
                for neighbor_id, rel_type, weight in self.neighbors[entity_id]:
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
            if seed_id not in self.neighbors:
                continue
            seed_entity = self.id_to_entity.get(seed_id)
            if not seed_entity:
                continue

            for neighbor_id, rel_type, weight in self.neighbors[seed_id]:
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
        """Validate a specific claim."""
        subj_ids = self.name_to_ids.get(subject.lower(), [])
        obj_ids = self.name_to_ids.get(object.lower(), [])

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
                if subj_id in self.neighbors:
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
        n_relations = len(self.relations)
        dim = self.embeddings.shape[1] // 4 if self.embeddings is not None else 0

        file_size = 0.0
        if self._loaded_path and self._loaded_path.exists():
            file_size = self._loaded_path.stat().st_size / (1024 * 1024)

        if self.truth_weights is not None and len(self.truth_weights) > 0:
            high = 100 * (self.truth_weights > 0.7).sum() / len(self.truth_weights)
            medium = 100 * ((self.truth_weights > 0.3) & (self.truth_weights <= 0.7)).sum() / len(self.truth_weights)
            low = 100 * (self.truth_weights <= 0.3).sum() / len(self.truth_weights)
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
        """Run a data quality audit."""
        issues = []
        recommendations = []

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
        """Get the top hub entities by degree."""
        degree_counts = {}
        for eid in self.neighbors:
            degree_counts[eid] = len(self.neighbors[eid])

        top_ids = sorted(degree_counts.keys(), key=lambda x: -degree_counts[x])[:n]
        threshold = self._weight_threshold or 0.5

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

    def get_relations(self, entity_name: str) -> List[Any]:
        """Get all relations for an entity."""
        entity_id = self.name_to_id.get(entity_name.lower())
        if entity_id is None:
            return []

        @dataclass
        class RelInfo:
            relation: str
            target: str
            truth_weight: float

        results = []
        if entity_id in self.neighbors:
            for neighbor_id, rel_type, weight in self.neighbors[entity_id]:
                neighbor = self.id_to_entity.get(neighbor_id)
                rel_name = self.relation_type_names.get(rel_type, f"relation_{rel_type}")
                results.append(RelInfo(
                    relation=rel_name,
                    target=neighbor.name if neighbor else f"entity_{neighbor_id}",
                    truth_weight=weight
                ))

        return sorted(results, key=lambda r: -r.truth_weight)
