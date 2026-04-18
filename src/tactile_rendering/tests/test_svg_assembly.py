import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tactile.ir import load_and_validate
from tactile.layout import compute_layout
from tactile.svg_assembly import assemble

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "v1_simple_flowchart.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def test_assemble_returns_three_parts():
    fc = load_and_validate(FIXTURE_PATH)
    layout = compute_layout(fc)
    dwg, labels, sidecar = assemble(fc, layout)
    assert dwg is not None
    assert len(labels) == 5  # 5 nodes, no connection labels
    assert "legend" in sidecar


def test_sidecar_has_all_legend_entries():
    fc = load_and_validate(FIXTURE_PATH)
    layout = compute_layout(fc)
    _, _, sidecar = assemble(fc, layout)
    assert len(sidecar["legend"]) == 5
    assert sidecar["legend"][0]["print_text"] == "Computer"
    assert sidecar["legend"][0]["number"] == 1


def test_svg_written_to_file():
    fc = load_and_validate(FIXTURE_PATH)
    layout = compute_layout(fc)
    dwg, labels, sidecar = assemble(fc, layout)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    svg_path = OUTPUT_DIR / "m5_assembled.svg"
    dwg.saveas(str(svg_path), pretty=True)
    assert svg_path.exists()
    assert svg_path.stat().st_size > 0

    json_path = OUTPUT_DIR / "m5_tactile.json"
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    assert json_path.exists()
