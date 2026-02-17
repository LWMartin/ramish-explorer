"""
Lightweight data models for .ramish file entities and relations.

These are standalone copies of the Entity/Relation dataclasses from the
proprietary pipeline, containing only what the reader needs. No training
code, no pipeline dependencies.
"""

from dataclasses import dataclass


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    id: int
    external_id: str
    name: str
    entity_type: str


@dataclass
class Relation:
    """A relation (edge) in the knowledge graph."""
    head_id: int
    relation_type: int
    tail_id: int
    source_table: str = ""
