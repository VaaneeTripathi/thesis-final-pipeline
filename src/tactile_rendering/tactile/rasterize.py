"""Stage 6: Pin-grid rasterization.

Renders the structural SVG to a high-resolution PNG, then downsamples
to a binary pin grid (default 60×40) simulating a refreshable tactile
display.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image

# Default pin grid dimensions (Graphiti-like).
DEFAULT_GRID_COLS = 60
DEFAULT_GRID_ROWS = 40

# DPI for the intermediate high-resolution render.
# Higher = more accurate sampling; 300 is sufficient.
RENDER_DPI = 300

# Ink threshold: a pin turns on if the proportion of dark pixels
# in its cell exceeds this value (0–1).
INK_THRESHOLD = 0.15


def _svg_to_grayscale(svg_string: str) -> np.ndarray:
    """Render SVG to a grayscale numpy array via cairosvg.

    The SVG has no background fill, so cairosvg produces an RGBA image
    with transparent background. We composite onto white before
    converting to grayscale so that empty space = 255 (white) and
    ink = 0 (black).
    """
    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        dpi=RENDER_DPI,
    )
    rgba = Image.open(BytesIO(png_bytes)).convert("RGBA")
    white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white_bg, rgba)
    gray = composited.convert("L")
    return np.array(gray)


def rasterize(
    svg_string: str,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
) -> np.ndarray:
    """Rasterize an SVG string to a binary pin grid.

    Returns a numpy array of shape (grid_rows, grid_cols) with values
    0 (pin down) or 1 (pin up).
    """
    gray = _svg_to_grayscale(svg_string)
    img_h, img_w = gray.shape

    # Compute cell size for each pin.
    cell_w = img_w / grid_cols
    cell_h = img_h / grid_rows

    grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Pixel region for this pin.
            y0 = int(row * cell_h)
            y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w)
            x1 = int((col + 1) * cell_w)

            # Clamp bounds.
            y1 = min(y1, img_h)
            x1 = min(x1, img_w)

            if y0 >= y1 or x0 >= x1:
                continue

            cell = gray[y0:y1, x0:x1]

            # Dark pixels are ink (grayscale < 128 on a white background).
            dark_ratio = np.mean(cell < 128)
            if dark_ratio > INK_THRESHOLD:
                grid[row, col] = 1

    return grid


def grid_to_debug_image(
    grid: np.ndarray,
    scale: int = 10,
) -> Image.Image:
    """Convert a pin grid to a debug PNG image.

    Each pin is rendered as a circle on a scaled-up canvas.
    Pin up = black circle, pin down = white.
    """
    rows, cols = grid.shape
    img_w = cols * scale
    img_h = rows * scale

    img = Image.new("RGB", (img_w, img_h), "white")
    pixels = img.load()

    radius = scale // 3

    for row in range(rows):
        for col in range(cols):
            cx = col * scale + scale // 2
            cy = row * scale + scale // 2

            if grid[row, col] == 1:
                # Draw a filled circle.
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx * dx + dy * dy <= radius * radius:
                            px = cx + dx
                            py = cy + dy
                            if 0 <= px < img_w and 0 <= py < img_h:
                                pixels[px, py] = (0, 0, 0)
            else:
                # Draw a faint dot to show the grid.
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        px = cx + dx
                        py = cy + dy
                        if 0 <= px < img_w and 0 <= py < img_h:
                            pixels[px, py] = (220, 220, 220)

    return img
