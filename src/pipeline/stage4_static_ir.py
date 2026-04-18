"""Stage 4: Static IR generation (triggered by Mealy S output).

Merges CV geometry (ElementRegistry — positions, bboxes, CV-detected shapes)
with VLM semantics (BoardSnapshot — text, shape verification, connections,
hierarchy, symbol meanings, annotations) to produce a dict conforming to
static-schema.json.

Design contract:
  - CV registry  = ground truth for geometry (positions, bounding boxes)
  - BoardSnapshot = ground truth for semantics (text, shapes, connections, meaning)
  - Where they disagree on shape: VLM wins (it sees context); discrepancy is logged
  - Where VLM references a mark not in registry: skip it (hallucination guard)

Node IDs in the output are clean semantic identifiers (n1, n2, ...) assigned
by reading-order position (top-to-bottom, left-to-right). SoM mark IDs are
internal only — they never appear in the final IR.
"""
from __future__ import annotations
import copy
import datetime
import logging
from typing import Any

from pipeline.models import BoardSnapshot, ElementRegistry

log = logging.getLogger(__name__)

# Canonical shape map — values must match the static-schema.json shape enum exactly.
_SHAPE_MAP: dict[str, str] = {
    "rectangle": "rectangle",
    "diamond": "diamond",
    "oval": "oval",
    "parallelogram": "parallelogram",
    "circle": "circle",
    "rounded-rectangle": "rounded-rectangle",
    "triangle": "triangle",
    "other": "other",
}

# CV shape types that indicate a connection/edge, not a node.
_CONNECTION_SHAPES = {"arrow", "line", "connection"}

# Annotation types recognised by the schema.
_ANNOTATION_TYPES = {
    "container", "separator", "highlight", "circle",
    "underline", "arrow", "cloud", "strikethrough",
}


def _assign_node_ids(
    registry: ElementRegistry,
    mark_descriptions: dict[int, dict[str, Any]],
) -> dict[int, str]:
    """Return a mapping mark_id → nX using top-to-bottom, left-to-right order.

    Nodes are sorted by centroid (y bucketed into rows of 50 px, then x)
    to follow the left-to-right, level-by-level convention.
    Marks that CV or VLM classifies as connections or annotations are excluded.
    """
    node_regions = []
    for mark_id, region in registry.elements.items():
        if region.shape_type in _CONNECTION_SHAPES:
            continue
        vlm_type = mark_descriptions.get(mark_id, {}).get("element_type", "node")
        if vlm_type in {"connection", "annotation"}:
            continue
        node_regions.append(region)

    node_regions.sort(key=lambda r: (r.centroid[1] // 50, r.centroid[0]))
    return {r.mark_id: f"n{i + 1}" for i, r in enumerate(node_regions)}


def build(
    registry: ElementRegistry,
    snapshot: BoardSnapshot,
    time_taken: str = "00:00",
) -> dict:
    """Build a static-schema.json-conformant dict from CV geometry + VLM snapshot.

    Args:
        registry:   ElementRegistry at the pen-lift moment (CV ground truth).
        snapshot:   BoardSnapshot from stage2.analyse_snapshots() for this keyframe.
        time_taken: Wall-clock analysis duration in MM:SS.

    Returns:
        dict conforming to static-schema.json.
        Node IDs are n1, n2, ... — no SoM mark IDs appear in the output.
    """
    mark_descriptions = snapshot.mark_descriptions
    mark_to_nid = _assign_node_ids(registry, mark_descriptions)
    valid_mark_ids = set(mark_to_nid.keys())

    
    for mid in mark_descriptions:
        if mid not in registry.elements:
            log.warning(
                "VLM described mark [%d] not present in CV registry — skipping (possible hallucination)",
                mid,
            )

   
    nodes: list[dict] = []
    for mark_id in sorted(mark_to_nid.keys()):
        region = registry.elements[mark_id]
        desc = mark_descriptions.get(mark_id, {})
        x, y, w, h = region.bbox

        # Shape: prefer VLM (semantic context) over CV (geometric heuristic)
        cv_shape = _SHAPE_MAP.get(region.shape_type, "other")
        vlm_shape_raw = desc.get("shape")
        vlm_shape = _SHAPE_MAP.get(vlm_shape_raw, "other") if vlm_shape_raw else None

        if vlm_shape and vlm_shape != cv_shape:
            log.debug(
                "Shape mismatch mark [%d]: CV=%s VLM=%s — using VLM",
                mark_id, cv_shape, vlm_shape,
            )
        shape = vlm_shape if vlm_shape else cv_shape

        node: dict = {
            "id": mark_to_nid[mark_id],
            "shape": shape,
            "text": desc.get("text"),
            # Position from CV registry — always ground truth
            "position": {"x": float(x + w // 2), "y": float(y + h // 2)},
        }

        # Only include visual if VLM found a meaningful color
        color = (desc.get("visual") or {}).get("color")
        if color:
            node["visual"] = {"color": color}

        nodes.append(node)

    
    connections: list[dict] = []
    conn_idx = 1
    for conn in snapshot.connections:
        from_id = conn.get("from_mark")
        to_id = conn.get("to_mark")

        if from_id not in valid_mark_ids or to_id not in valid_mark_ids:
            log.debug(
                "Skipping dangling connection [%s]->[%s]: mark not in node registry",
                from_id, to_id,
            )
            continue

        direction = conn.get("direction", "forward")
        if direction not in {"forward", "backward", "bidirectional", "none"}:
            direction = "forward"

        line_type = conn.get("line_type", "solid")
        if line_type not in {"solid", "dashed", "dotted"}:
            line_type = "solid"

        entry: dict = {
            "id": f"c{conn_idx}",
            "source": mark_to_nid[from_id],
            "target": mark_to_nid[to_id],
            "direction": direction,
            "line_type": line_type,
        }
        label = conn.get("label")
        if label:
            entry["label"] = label
        connections.append(entry)
        conn_idx += 1

    
    
    hierarchy: list[dict] = []
    for i, group in enumerate(snapshot.groupings):
        label = group.get("label") or f"group-{i + 1}"
        members = [
            mark_to_nid[m]
            for m in group.get("members", [])
            if m in mark_to_nid
        ]
        if not members:
            continue
        hierarchy.append({
            "id": label,
            "label": label,
            "parent": group.get("parent"),
            "children": members,
        })

    
    present_shapes = {
        (mark_descriptions.get(mid, {}).get("shape") or
         _SHAPE_MAP.get(registry.elements[mid].shape_type, "other"))
        for mid in valid_mark_ids
    }
    symbols: list[dict] = []
    for sm in snapshot.symbol_meanings:
        shape = sm.get("shape")
        meaning = sm.get("meaning")
        if not shape or not meaning:
            continue
        if shape not in present_shapes:
            continue
        applicable = [
            mark_to_nid[mid] for mid in valid_mark_ids
            if (mark_descriptions.get(mid, {}).get("shape") or
                _SHAPE_MAP.get(registry.elements[mid].shape_type, "other")) == shape
        ]
        symbols.append({
            "symbol": shape,
            "meaning": meaning,
            "applicable_to": applicable,
        })

    
    annotations: list[dict] = []
    ann_idx = 1
    for ann in snapshot.annotations:
        mark_id = ann.get("mark_id")
        ann_type = ann.get("annotation_type", "highlight")
        if mark_id not in mark_to_nid:
            log.debug("Skipping annotation on unknown mark [%s]", mark_id)
            continue
        if ann_type not in _ANNOTATION_TYPES:
            ann_type = "highlight"
        annotations.append({
            "id": f"ann-{ann_idx}",
            "target": mark_to_nid[mark_id],
            "annotation_type": ann_type,
            "content": ann.get("content"),
        })
        ann_idx += 1

    
    cross_links: list[dict] = []
    for cl in snapshot.cross_links:
        if all(cl.get(k) for k in ("source_flowchart", "target_flowchart",
                                    "source_element", "target_element")):
            cross_links.append(copy.copy(cl))

    
    structure: dict = {}
    if hierarchy:
        structure["hierarchy"] = hierarchy

    state: dict = {
        "board_state": snapshot.board_state or (
            f"Board at {snapshot.timestamp:.1f}s: {len(nodes)} elements visible."
        ),
    }
    if cross_links:
        state["cross_links"] = cross_links

    semantics: dict = {}
    if symbols:
        semantics["symbols"] = symbols
    if annotations:
        semantics["annotations"] = annotations

    return {
        "id": datetime.datetime.utcnow().isoformat() + "Z",
        "elements": {
            "nodes": nodes,
            "connections": connections,
        },
        "structure": structure,
        "state": state,
        "semantics": semantics,
        "provenance": {
            "model": "gemini-2.5-flash + opencv-cv",
            "confidence": snapshot.confidence,
            "visibility_issues": snapshot.visibility_issues,
            "time_taken": time_taken,
        },
    }
