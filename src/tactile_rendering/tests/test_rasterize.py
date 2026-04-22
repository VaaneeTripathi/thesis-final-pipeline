import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tactile.ir import load_and_validate
from tactile.layout import compute_layout
from tactile.svg_assembly import assemble
from tactile.rasterize import rasterize, grid_to_debug_image

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "v1_simple_flowchart.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _get_svg_string():
    fc = load_and_validate(FIXTURE_PATH)
    layout = compute_layout(fc)
    dwg, _, _ = assemble(fc, layout)
    return dwg.tostring()


def test_rasterize_returns_grid():
    svg = _get_svg_string()
    grid = rasterize(svg)
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (40, 60)
    assert grid.dtype == np.uint8


def test_grid_has_some_pins_up():
    svg = _get_svg_string()
    grid = rasterize(svg)
    # A 5-node flowchart should have *some* pins raised.
    assert np.sum(grid) > 0


def test_grid_is_not_all_on():
    svg = _get_svg_string()
    grid = rasterize(svg)
    # Should not be nearly solid black — that would mean thresholding is broken.
    # Larger BANA-compliant nodes legitimately fill more of the grid, so we
    # only fail if > 90% of pins are raised (which would indicate a rendering error).
    total = grid.shape[0] * grid.shape[1]
    assert np.sum(grid) < total * 0.9


def test_save_outputs():
    svg = _get_svg_string()
    grid = rasterize(svg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the numpy grid.
    npy_path = OUTPUT_DIR / "m7_grid.npy"
    np.save(str(npy_path), grid)
    assert npy_path.exists()

    # Save the debug PNG.
    debug_img = grid_to_debug_image(grid)
    png_path = OUTPUT_DIR / "m7_grid.png"
    debug_img.save(str(png_path))
    assert png_path.exists()
    assert png_path.stat().st_size > 0
