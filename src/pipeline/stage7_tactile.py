"""Stage 7: Tactile rendering bridge.

For each validated static IR snapshot, feeds the JSON into the tactile
rendering pipeline (tactile-rendering/src/tactile/pipeline.py) to produce:
  - diagram.svg           — structural SVG with shapes, edges, numeric IDs
  - key_page_NNNN.svg     — numbered Braille key (one file per page)
  - tactile.json          — sidecar: key table, Braille, BANA metadata
  - grid.npy              — 60×40 binary pin-grid for refreshable tactile display
  - grid.png              — debug visualisation of the pin grid
  - pin_delta.npy         — int8 diff vs previous snapshot (+1 raise, -1 lower, 0 none)
  - pin_delta.png         — debug: green = raised, red = lowered, grey = unchanged
  - pin_delta_meta.json   — counts of changed pins; use to decide partial vs full refresh

Outputs are written to output_dir/tactile/snapshot_NNNN/ per snapshot.

The tactile renderer lives in a sibling repo (tactile-rendering/) and is
imported at runtime by inserting its src/ directory into sys.path. If the
package is not available the stage logs a warning and returns gracefully —
the rest of the pipeline is unaffected.
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# src/                        ← _PIPELINE_SRC  (contains pipeline package)
# src/tactile_rendering/      ← _TACTILE_SRC   (contains tactile package)
_PIPELINE_SRC = Path(__file__).resolve().parent.parent          # .../src
_TACTILE_SRC  = _PIPELINE_SRC / "tactile_rendering"             # .../src/tactile_rendering


def _import_tactile_pipeline():
    """Attempt to import tactile.pipeline, returning the module or None.

    tactile/pipeline.py uses bare ``from tactile.ir import ...`` imports,
    so ``src/tactile_rendering/`` must be on sys.path (not just ``src/``).
    """
    for p in (str(_PIPELINE_SRC), str(_TACTILE_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from tactile import pipeline as tp
        return tp
    except ImportError as exc:
        log.warning(
            "Stage 7: tactile renderer not available (%s) — skipping tactile output. "
            "Check that %s exists and its dependencies are installed.",
            exc, _TACTILE_SRC,
        )
        return None


def _compute_pin_delta(prev_grid: np.ndarray, new_grid: np.ndarray) -> np.ndarray:
    """Return int8 array: +1 = pin raised, -1 = pin lowered, 0 = unchanged."""
    delta = np.zeros_like(new_grid, dtype=np.int8)
    delta[new_grid > prev_grid] = 1
    delta[new_grid < prev_grid] = -1
    return delta


def _save_pin_delta_debug(delta: np.ndarray, path: Path) -> None:
    """Save a debug PNG visualising the pin delta.

    Colour scheme:
      green  (#00b400) = pin raised   (0 → 1)
      red    (#c80000) = pin lowered  (1 → 0)
      grey   (#c8c8c8) = unchanged & unpinned
      black  (#000000) = unchanged & pinned
    """
    try:
        from PIL import Image
    except ImportError:
        log.debug("PIL not available — skipping pin_delta.png")
        return

    h, w = delta.shape
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)   # default: grey (unchanged-off)
    rgb[delta == 1]  = [0, 180, 0]                   # green  = raised
    rgb[delta == -1] = [200, 0, 0]                   # red    = lowered
    # Note: unchanged-on pins stay grey; we don't have the full grid here.
    # The full grid debug image (grid.png) already shows the absolute state.
    img = Image.fromarray(rgb).resize((w * 12, h * 12), Image.NEAREST)
    img.save(str(path))


def run(output_dir: Path) -> list[Path]:
    """Render each snapshot in output_dir/snapshots/ through the tactile pipeline.

    Args:
        output_dir: Pipeline root output directory. Must already contain
                    snapshots/snapshot_NNNN.json files from stage 6.

    Returns:
        List of tactile output directories that were successfully rendered,
        in snapshot order. Empty list if tactile renderer is unavailable or
        no snapshots exist.
    """
    tactile_mod = _import_tactile_pipeline()
    if tactile_mod is None:
        return []

    snapshots_dir = output_dir / "snapshots"
    if not snapshots_dir.exists():
        log.warning("Stage 7: no snapshots/ directory found in %s — nothing to render", output_dir)
        return []

    snapshot_paths = sorted(snapshots_dir.glob("snapshot_*.json"))
    if not snapshot_paths:
        log.info("Stage 7: no snapshot files found — nothing to render")
        return []

    tactile_root = output_dir / "tactile"
    tactile_root.mkdir(parents=True, exist_ok=True)

    successful: list[Path] = []
    # Tracks the pin state currently "on the display". Starts as a blank board
    # (all pins down). Updated only on successful renders.
    prev_grid: np.ndarray | None = None

    for snap_path in snapshot_paths:
        # Derive index from filename: snapshot_0003.json → "0003"
        stem = snap_path.stem  # e.g. "snapshot_0003"
        idx = stem.split("_", 1)[1] if "_" in stem else stem
        out_dir = tactile_root / f"snapshot_{idx}"

        log.info("Stage 7: rendering %s → %s", snap_path.name, out_dir)
        try:
            tactile_mod.run(snap_path, out_dir)
        except Exception as exc:
            log.error(
                "Stage 7: tactile rendering failed for %s: %s — continuing",
                snap_path.name, exc,
            )
            continue

        # Pin-level delta
        grid_path = out_dir / "grid.npy"
        if not grid_path.exists():
            log.warning("Stage 7: grid.npy missing in %s — skipping pin delta", out_dir)
            successful.append(out_dir)
            continue

        new_grid = np.load(str(grid_path))

        # Treat a blank board as the baseline before the very first snapshot.
        if prev_grid is None:
            prev_grid = np.zeros_like(new_grid, dtype=np.uint8)

        delta = _compute_pin_delta(prev_grid, new_grid)

        np.save(str(out_dir / "pin_delta.npy"), delta)
        _save_pin_delta_debug(delta, out_dir / "pin_delta.png")

        pins_raised  = int(np.sum(delta == 1))
        pins_lowered = int(np.sum(delta == -1))
        pins_total   = int(delta.size)
        (out_dir / "pin_delta_meta.json").write_text(json.dumps({
            "from_snapshot": f"snapshot_{_prev_idx(idx, snapshot_paths)}",
            "to_snapshot":   f"snapshot_{idx}",
            "pins_raised":   pins_raised,
            "pins_lowered":  pins_lowered,
            "pins_changed":  pins_raised + pins_lowered,
            "pins_total":    pins_total,
            "pct_changed":   round((pins_raised + pins_lowered) / pins_total * 100, 2),
        }, indent=2))

        prev_grid = new_grid
        successful.append(out_dir)

    log.info(
        "Stage 7: %d/%d snapshots rendered successfully",
        len(successful), len(snapshot_paths),
    )
    return successful


def _prev_idx(current_idx: str, snapshot_paths: list[Path]) -> str:
    """Return the index string of the snapshot before current_idx, or 'blank'."""
    for i, p in enumerate(snapshot_paths):
        stem = p.stem
        idx = stem.split("_", 1)[1] if "_" in stem else stem
        if idx == current_idx:
            if i == 0:
                return "blank"
            prev_stem = snapshot_paths[i - 1].stem
            return prev_stem.split("_", 1)[1] if "_" in prev_stem else prev_stem
    return "unknown"
