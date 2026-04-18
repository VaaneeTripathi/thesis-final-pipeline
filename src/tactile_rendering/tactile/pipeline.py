"""End-to-end tactile rendering pipeline.

Usage:
    python -m tactile.pipeline <ir_json_path> <output_dir>

Runs all six stages and writes outputs to the specified directory:
    - diagram.svg       (structural SVG with shapes, edges, numeric IDs)
    - legend.svg        (numbered braille legend)
    - tactile.json      (sidecar: legend table, braille, BANA metadata)
    - grid.npy          (60×40 binary pin-grid matrix)
    - grid.png          (debug visualization of the pin grid)
    - layout_debug.svg  (raw Graphviz layout for visual inspection)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from tactile.ir import load_and_validate
from tactile.layout import compute_layout
from tactile.svg_assembly import assemble
from tactile.legend import render_legend
from tactile.rasterize import rasterize, grid_to_debug_image


def run(ir_path: str | Path, output_dir: str | Path) -> None:
    """Execute the full pipeline from IR JSON to tactile outputs."""
    ir_path = Path(ir_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Ingest and validate.
    print(f"[Stage 1] Loading {ir_path.name}...")
    flowchart = load_and_validate(ir_path)
    print(f"  {len(flowchart.nodes)} nodes, {len(flowchart.connections)} connections")
    if flowchart.warnings:
        for w in flowchart.warnings:
            print(f"  WARNING: {w}")

    # Stage 2: Layout via Graphviz.
    print("[Stage 2] Computing layout via Graphviz dot...")
    layout = compute_layout(flowchart)
    print(f"  Graph: {layout.graph_width:.2f} x {layout.graph_height:.2f} inches")

    # Save debug layout SVG.
    if hasattr(layout, "_debug_svg"):
        (output_dir / "layout_debug.svg").write_text(layout._debug_svg)

    # Stage 3 + 4: Braille transcription and SVG assembly.
    print("[Stage 3] Transcribing labels to UEB Grade 1 braille...")
    print("[Stage 4] Assembling structural SVG...")
    dwg, braille_labels, sidecar = assemble(flowchart, layout)

    # Write SVG.
    svg_path = output_dir / "diagram.svg"
    dwg.saveas(str(svg_path), pretty=True)
    print(f"  -> {svg_path}")

    # Write sidecar JSON.
    json_path = output_dir / "tactile.json"
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    print(f"  -> {json_path}")

    # Stage 5: Legend SVG.
    print("[Stage 5] Rendering legend...")
    legend_dwg = render_legend(braille_labels)
    legend_path = output_dir / "legend.svg"
    legend_dwg.saveas(str(legend_path), pretty=True)
    print(f"  -> {legend_path}")

    # Print legend table.
    print(f"\n  {'#':<4} {'Label':<12} {'Braille'}")
    print(f"  {'-' * 36}")
    for bl in braille_labels:
        print(f"  {bl.legend_number:<4} {bl.print_text:<12} {bl.braille_text}")

    # Stage 6: Pin-grid rasterization.
    print("\n[Stage 6] Rasterizing to 60x40 pin grid...")
    svg_string = dwg.tostring()
    grid = rasterize(svg_string)

    # Save grid.
    npy_path = output_dir / "grid.npy"
    np.save(str(npy_path), grid)
    print(f"  -> {npy_path}")

    # Save debug PNG.
    debug_img = grid_to_debug_image(grid)
    png_path = output_dir / "grid.png"
    debug_img.save(str(png_path))
    print(f"  -> {png_path}")

    # Summary.
    pins_up = int(np.sum(grid))
    total_pins = grid.shape[0] * grid.shape[1]
    print(f"\n  Pin grid: {grid.shape[1]}x{grid.shape[0]}, "
          f"{pins_up}/{total_pins} pins raised ({100 * pins_up / total_pins:.1f}%)")

    print("\nDone.")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m tactile.pipeline <ir_json_path> <output_dir>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
