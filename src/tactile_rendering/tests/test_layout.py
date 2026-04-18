import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tactile.ir import load_and_validate
from tactile.layout import compute_layout, LayoutResult, NodeLayout, EdgeLayout

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "v1_simple_flowchart.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def test_layout_returns_result():
    fc = load_and_validate(FIXTURE_PATH)
    layout = compute_layout(fc)
    assert isinstance(layout, LayoutResult)


def test_all_nodes_have_layout():
    fc = load_and_validate(FIXTURE_PATH)
    layout = compute_layout(fc)
    for node in fc.nodes:
        assert node.id in layout.nodes, f"Missing layout for {node.id}"
        nl = layout.nodes[node.id]
        assert isinstance(nl, NodeLayout)
        assert nl.width > 0
        assert nl.height > 0


def test_all_edges_have_layout():
    fc = load_and_validate(FIXTURE_PATH)
    layout = compute_layout(fc)
    assert len(layout.edges) == len(fc.connections)
    for edge in layout.edges:
        assert isinstance(edge, EdgeLayout)
        assert len(edge.points) >= 2


def test_graph_dimensions_positive():
    fc = load_and_validate(FIXTURE_PATH)
    layout = compute_layout(fc)
    assert layout.graph_width > 0
    assert layout.graph_height > 0


def test_debug_svg_saved():
    fc = load_and_validate(FIXTURE_PATH)
    layout = compute_layout(fc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIR / "m3_layout.svg"
    svg_path.write_text(layout._debug_svg)
    assert svg_path.exists()
    assert svg_path.stat().st_size > 0
