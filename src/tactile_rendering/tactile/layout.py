"""Stage 2: Layout via Graphviz.

Translates a Flowchart into a Graphviz graph, runs `dot` to compute
positions, parses the `plain` output format, and returns a LayoutResult
that downstream stages use for rendering.

BANA sizing (Guidelines and Standards for Tactile Graphics, 2022):
  - Minimum symbol size for discrimination: 1/4 inch (6 mm).
    Practical flowchart nodes must be larger to contain a legible number.
    We use 1.0" × 0.5" (25.4 × 12.7 mm) as the minimum.
  - Minimum separation between elements: 1/8 inch (3 mm).
    We use 0.5" (12.7 mm) between siblings and 0.6" (15.2 mm) between ranks.
  - Tactile grid unit: 0.25 inch (6.35 mm).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pydot

from tactile.ir import Flowchart

NODE_MIN_WIDTH_IN  = 1.0   # 25.4 mm — minimum to fit a 2-digit number
NODE_MIN_HEIGHT_IN = 0.5   # 12.7 mm

NODE_SEP_IN   = 0.5        # horizontal gap between sibling nodes (12.7 mm)
RANK_SEP_IN   = 0.6        # vertical gap between ranks (15.2 mm)
GRAPH_MARGIN_IN = 0.4      # border around the whole diagram (10.2 mm)


@dataclass
class NodeLayout:
    """Layout data for a single node, in inches (Graphviz native unit)."""
    id: str
    cx: float
    cy: float
    width: float
    height: float
    shape: str


@dataclass
class EdgeLayout:
    """Layout data for a single connection."""
    id: str
    source: str
    target: str
    direction: str
    points: list[tuple[float, float]]


@dataclass
class LayoutResult:
    """Complete layout for a flowchart."""
    nodes: dict[str, NodeLayout] = field(default_factory=dict)
    edges: list[EdgeLayout] = field(default_factory=list)
    graph_width: float = 0.0
    graph_height: float = 0.0


_SHAPE_MAP = {
    "rectangle":         "box",
    "rounded-rectangle": "box",      # style=rounded added separately
    "diamond":           "diamond",
    "oval":              "ellipse",
    "circle":            "circle",
    "parallelogram":     "parallelogram",
    "triangle":          "triangle",
    "other":             "box",
}


def _build_graphviz_graph(flowchart: Flowchart) -> pydot.Dot:
    graph = pydot.Dot(graph_type="digraph", rankdir="TB")

    graph.set_graph_defaults(
        nodesep=str(NODE_SEP_IN),
        ranksep=str(RANK_SEP_IN),
        margin=str(GRAPH_MARGIN_IN),
    )
    # fixedsize=false lets nodes grow beyond the minimum if a label is long,
    # but sets a floor so small nodes never drop below BANA minimums.
    graph.set_node_defaults(
        fontsize="10",
        margin="0.15,0.10",
        width=str(NODE_MIN_WIDTH_IN),
        height=str(NODE_MIN_HEIGHT_IN),
        fixedsize="false",
    )

    for node in flowchart.nodes:
        gv_shape = _SHAPE_MAP.get(node.shape, "box")
        label = node.text if node.text else node.id
        attrs = {
            "label": label,
            "shape": gv_shape,
            "width": str(NODE_MIN_WIDTH_IN),
            "height": str(NODE_MIN_HEIGHT_IN),
        }
        if node.shape == "rounded-rectangle":
            attrs["style"] = "rounded"
        graph.add_node(pydot.Node(node.id, **attrs))

    for conn in flowchart.connections:
        attrs: dict = {}
        if conn.direction == "none":
            attrs["dir"] = "none"
        elif conn.direction == "backward":
            attrs["dir"] = "back"
        elif conn.direction == "bidirectional":
            attrs["dir"] = "both"
        if conn.label:
            attrs["label"] = conn.label
        if conn.line_type == "dashed":
            attrs["style"] = "dashed"
        elif conn.line_type == "dotted":
            attrs["style"] = "dotted"
        graph.add_edge(pydot.Edge(conn.source, conn.target, **attrs))

    return graph


def _parse_plain_output(plain: str, flowchart: Flowchart) -> LayoutResult:
    """Parse Graphviz `plain` format into a LayoutResult.

    Plain format:
      graph SCALE WIDTH HEIGHT
      node  NAME X Y WIDTH HEIGHT LABEL STYLE SHAPE COLOR FILLCOLOR
      edge  TAIL HEAD N X1 Y1 ... XN YN [LABEL XL YL] STYLE COLOR
      stop
    """
    result = LayoutResult()

    conn_lookup: dict[tuple[str, str], list] = {}
    for conn in flowchart.connections:
        conn_lookup.setdefault((conn.source, conn.target), []).append(conn)

    # First pass: get graph height for Y-flip (plain is bottom-up, SVG top-down).
    graph_height = 0.0
    for line in plain.splitlines():
        tokens = line.strip().split()
        if tokens and tokens[0] == "graph":
            graph_height = float(tokens[3])
            break

    for line in plain.splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue

        if tokens[0] == "graph":
            result.graph_width  = float(tokens[2])
            result.graph_height = float(tokens[3])

        elif tokens[0] == "node":
            name = tokens[1]
            node = flowchart.node_by_id(name)
            result.nodes[name] = NodeLayout(
                id=name,
                cx=float(tokens[2]),
                cy=graph_height - float(tokens[3]),
                width=float(tokens[4]),
                height=float(tokens[5]),
                shape=node.shape,
            )

        elif tokens[0] == "edge":
            tail, head = tokens[1], tokens[2]
            n_pts = int(tokens[3])
            points, idx = [], 4
            for _ in range(n_pts):
                points.append((float(tokens[idx]), graph_height - float(tokens[idx + 1])))
                idx += 2

            conns = conn_lookup.get((tail, head), [])
            if conns:
                conn = conns.pop(0)
                edge_id, direction = conn.id, conn.direction
            else:
                edge_id, direction = f"e_{tail}_{head}", "forward"

            result.edges.append(EdgeLayout(
                id=edge_id, source=tail, target=head,
                direction=direction, points=points,
            ))

    return result


def compute_layout(flowchart: Flowchart) -> LayoutResult:
    """Run Graphviz dot on a Flowchart and return the computed layout."""
    graph = _build_graphviz_graph(flowchart)
    plain = graph.create(format="plain").decode("utf-8")
    result = _parse_plain_output(plain, flowchart)
    result._debug_svg = graph.create(format="svg").decode("utf-8")
    return result
