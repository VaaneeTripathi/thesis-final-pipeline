"""Stage 1: IR ingest and validation.

Loads a static-schema.json instance, validates it against the schema,
and parses it into typed dataclasses for downstream pipeline stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator, validate

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "static-schema.json"

# Features that v1 does not handle — presence triggers a warning, not an error.
_V1_UNSUPPORTED_WARNINGS = {
    ("state", "cross_links"): "Cross-links are not rendered in v1.",
    ("state", "compacted"): "Compacted view is not rendered in v1.",
    ("semantics", "symbols"): "Symbols are not rendered in v1.",
    ("semantics", "annotations"): "Annotations are not rendered in v1.",
}


@dataclass
class Position:
    x: float
    y: float


@dataclass
class Node:
    id: str
    shape: str
    text: str | None
    position: Position | None = None
    color: str | None = None


@dataclass
class Connection:
    id: str
    source: str
    target: str
    direction: str = "forward"
    line_type: str = "solid"
    label: str | None = None
    color: str | None = None


@dataclass
class HierarchyGroup:
    id: str
    parent: str | None
    children: list[str]
    label: str | None = None


@dataclass
class Provenance:
    model: str
    confidence: str
    visibility_issues: str | None
    time_taken: str
    reasoning_steps: int | None = None


@dataclass
class Flowchart:
    """Typed in-memory representation of a static IR instance."""

    id: str
    nodes: list[Node]
    connections: list[Connection]
    hierarchy: list[HierarchyGroup] = field(default_factory=list)
    board_state: str | None = None
    provenance: Provenance | None = None
    warnings: list[str] = field(default_factory=list)

    def node_by_id(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"No node with id '{node_id}'")


def _parse_position(raw: dict | None) -> Position | None:
    if raw is None:
        return None
    return Position(x=raw["x"], y=raw["y"])


def _parse_visual_color(raw: dict | None) -> str | None:
    if raw is None:
        return None
    return raw.get("color")


def _parse_node(raw: dict) -> Node:
    return Node(
        id=raw["id"],
        shape=raw["shape"],
        text=raw.get("text"),
        position=_parse_position(raw.get("position")),
        color=_parse_visual_color(raw.get("visual")),
    )


def _parse_connection(raw: dict) -> Connection:
    return Connection(
        id=raw["id"],
        source=raw["source"],
        target=raw["target"],
        direction=raw.get("direction", "forward"),
        line_type=raw.get("line_type", "solid"),
        label=raw.get("label"),
        color=_parse_visual_color(raw.get("visual")),
    )


def _parse_hierarchy(raw_structure: dict) -> list[HierarchyGroup]:
    groups_raw = raw_structure.get("hierarchy")
    if not groups_raw:
        return []
    return [
        HierarchyGroup(
            id=g["id"],
            parent=g.get("parent"),
            children=g["children"],
            label=g.get("label"),
        )
        for g in groups_raw
    ]


def _parse_provenance(raw: dict) -> Provenance:
    return Provenance(
        model=raw["model"],
        confidence=raw["confidence"],
        visibility_issues=raw.get("visibility_issues"),
        time_taken=raw["time_taken"],
        reasoning_steps=raw.get("reasoning_steps"),
    )


def _collect_warnings(raw: dict) -> list[str]:
    warnings = []
    for (section, key), msg in _V1_UNSUPPORTED_WARNINGS.items():
        value = raw.get(section, {}).get(key)
        if value is not None and value != [] and value != {}:
            warnings.append(f"{section}.{key}: {msg}")
    return warnings


def load_and_validate(ir_path: str | Path, schema_path: str | Path | None = None) -> Flowchart:
    """Load a static IR JSON file, validate it, and return a typed Flowchart."""
    ir_path = Path(ir_path)
    schema_path = Path(schema_path) if schema_path else SCHEMA_PATH

    raw = json.loads(ir_path.read_text())
    schema = json.loads(schema_path.read_text())

    Draft202012Validator.check_schema(schema)
    validate(instance=raw, schema=schema, cls=Draft202012Validator)

    warnings = _collect_warnings(raw)

    elements = raw["elements"]
    nodes = [_parse_node(n) for n in elements["nodes"]]
    connections = [_parse_connection(c) for c in elements["connections"]]
    hierarchy = _parse_hierarchy(raw.get("structure", {}))
    board_state = raw.get("state", {}).get("board_state")
    provenance = _parse_provenance(raw["provenance"]) if "provenance" in raw else None

    return Flowchart(
        id=raw["id"],
        nodes=nodes,
        connections=connections,
        hierarchy=hierarchy,
        board_state=board_state,
        provenance=provenance,
        warnings=warnings,
    )
