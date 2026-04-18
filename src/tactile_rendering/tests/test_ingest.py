import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tactile.ir import load_and_validate, Flowchart, Node, Connection

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "v1_simple_flowchart.json"


def test_load_returns_flowchart():
    fc = load_and_validate(FIXTURE_PATH)
    assert isinstance(fc, Flowchart)


def test_nodes_parsed():
    fc = load_and_validate(FIXTURE_PATH)
    assert len(fc.nodes) == 5
    assert all(isinstance(n, Node) for n in fc.nodes)
    assert fc.nodes[0].id == "n1"
    assert fc.nodes[0].shape == "rectangle"
    assert fc.nodes[0].text == "Computer"


def test_connections_parsed():
    fc = load_and_validate(FIXTURE_PATH)
    assert len(fc.connections) == 4
    assert all(isinstance(c, Connection) for c in fc.connections)
    assert fc.connections[0].source == "n1"
    assert fc.connections[0].target == "n2"
    assert fc.connections[0].direction == "forward"


def test_hierarchy_parsed():
    fc = load_and_validate(FIXTURE_PATH)
    assert len(fc.hierarchy) == 1
    assert fc.hierarchy[0].parent is None
    assert "n1" in fc.hierarchy[0].children


def test_provenance_parsed():
    fc = load_and_validate(FIXTURE_PATH)
    assert fc.provenance is not None
    assert fc.provenance.model == "hand-authored"
    assert fc.provenance.confidence == "high"


def test_positions_parsed():
    fc = load_and_validate(FIXTURE_PATH)
    n1 = fc.node_by_id("n1")
    assert n1.position is not None
    assert n1.position.x == 0
    assert n1.position.y == 0


def test_node_by_id_raises_on_missing():
    fc = load_and_validate(FIXTURE_PATH)
    try:
        fc.node_by_id("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_no_warnings_for_v1_fixture():
    fc = load_and_validate(FIXTURE_PATH)
    assert fc.warnings == []
