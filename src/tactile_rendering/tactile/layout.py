"""Stage 2: Layout via Graphviz.

Translates a Flowchart into a Graphviz graph, runs `dot` to compute
positions, parses the `plain` output format, and returns a LayoutResult
that downstream stages use for rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pydot

from tactile.ir import Flowchart


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


# Map IR shape names to Graphviz shape names.
_SHAPE_MAP = {
    "rectangle": "box",
    "rounded-rectangle": "box",       # style=rounded added separately
    "diamond": "diamond",
    "oval": "ellipse",
    "circle": "circle",
    "parallelogram": "parallelogram",
    "triangle": "triangle",
    "other": "box",
}


def _build_graphviz_graph(flowchart: Flowchart) -> pydot.Dot:
    """Convert a Flowchart to a pydot graph object."""
    graph = pydot.Dot(graph_type="digraph", rankdir="TB")
    graph.set_node_defaults(fontsize="10", margin="0.2,0.1")

    for node in flowchart.nodes:
        gv_shape = _SHAPE_MAP.get(node.shape, "box")
        label = node.text if node.text else node.id
        attrs = {"label": label, "shape": gv_shape}
        if node.shape == "rounded-rectangle":
            attrs["style"] = "rounded"
        graph.add_node(pydot.Node(node.id, **attrs))

    for conn in flowchart.connections:
        attrs = {}
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

    Plain format reference:
      graph SCALE WIDTH HEIGHT
      node NAME X Y WIDTH HEIGHT LABEL STYLE SHAPE COLOR FILLCOLOR
      edge TAIL HEAD N X1 Y1 ... XN YN [LABEL XL YL] STYLE COLOR
      stop
    """
    result = LayoutResult()

    # Build a lookup from connection source-target to connection object,
    # so we can recover the connection id and direction from edge lines.
    conn_lookup: dict[tuple[str, str], list] = {}
    for conn in flowchart.connections:
        key = (conn.source, conn.target)
        conn_lookup.setdefault(key, []).append(conn)

    # First pass: read graph height for Y-axis flip.
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
            result.graph_width = float(tokens[2])
            result.graph_height = float(tokens[3])

        elif tokens[0] == "node":
            name = tokens[1]
            node = flowchart.node_by_id(name)
            # Flip Y: plain format is bottom-up, SVG is top-down.
            result.nodes[name] = NodeLayout(
                id=name,
                cx=float(tokens[2]),
                cy=graph_height - float(tokens[3]),
                width=float(tokens[4]),
                height=float(tokens[5]),
                shape=node.shape,
            )

        elif tokens[0] == "edge":
            tail = tokens[1]
            head = tokens[2]
            n_points = int(tokens[3])
            points = []
            idx = 4
            for _ in range(n_points):
                px = float(tokens[idx])
                # Flip Y for edge points too.
                py = graph_height - float(tokens[idx + 1])
                points.append((px, py))
                idx += 2

            conns = conn_lookup.get((tail, head), [])
            if conns:
                conn = conns.pop(0)
                edge_id = conn.id
                direction = conn.direction
            else:
                edge_id = f"e_{tail}_{head}"
                direction = "forward"

            result.edges.append(EdgeLayout(
                id=edge_id,
                source=tail,
                target=head,
                direction=direction,
                points=points,
            ))

    return result


def compute_layout(flowchart: Flowchart) -> LayoutResult:
    """Run Graphviz dot on a Flowchart and return the computed layout."""
    graph = _build_graphviz_graph(flowchart)

    # Get the plain-format output for structured parsing.
    plain_bytes = graph.create(format="plain")
    plain = plain_bytes.decode("utf-8")

    result = _parse_plain_output(plain, flowchart)

    # Also save the raw dot SVG as a debug attribute.
    result._debug_svg = graph.create(format="svg").decode("utf-8")

    return result
