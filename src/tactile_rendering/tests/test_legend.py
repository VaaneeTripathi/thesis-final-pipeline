import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tactile.braille import transcribe_labels
from tactile.legend import render_legend

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def test_legend_renders():
    labels = transcribe_labels([
        ("n1", "Computer"),
        ("n2", "Screen"),
        ("n3", "Keyboard"),
        ("n4", "Mouse"),
        ("n5", "Speakers"),
    ])
    dwg = render_legend(labels)
    assert dwg is not None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIR / "m6_legend.svg"
    dwg.saveas(str(svg_path), pretty=True)
    assert svg_path.exists()
    assert svg_path.stat().st_size > 0
