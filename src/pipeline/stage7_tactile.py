"""Stage 7: Tactile rendering bridge.

For each validated static IR snapshot, feeds the JSON into the tactile
rendering pipeline (tactile-rendering/src/tactile/pipeline.py) to produce:
  - diagram.svg       — structural SVG with shapes, edges, numeric IDs
  - legend.svg        — numbered Braille legend
  - tactile.json      — sidecar: legend table, Braille, BANA metadata
  - grid.npy          — 60×40 binary pin-grid for refreshable tactile display
  - grid.png          — debug visualisation of the pin grid

Outputs are written to output_dir/tactile/snapshot_NNNN/ per snapshot.

The tactile renderer lives in a sibling repo (tactile-rendering/) and is
imported at runtime by inserting its src/ directory into sys.path. If the
package is not available the stage logs a warning and returns gracefully —
the rest of the pipeline is unaffected.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_TACTILE_SRC = Path(__file__).resolve().parent.parent



def _import_tactile_pipeline():
    """Attempt to import tactile.pipeline, returning the module or None."""
    if str(_TACTILE_SRC) not in sys.path:
        sys.path.insert(0, str(_TACTILE_SRC))
    try:
        
        from tactile_rendering.tactile import pipeline as tp
        return tp
    except ImportError as exc:
        log.warning(
            "Stage 7: tactile renderer not available (%s) — skipping tactile output. "
            "Install tactile-rendering dependencies or check that %s exists.",
            exc, _TACTILE_SRC,
        )
        return None


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

    for snap_path in snapshot_paths:
        # Derive index from filename: snapshot_0003.json → "0003"
        stem = snap_path.stem  # e.g. "snapshot_0003"
        idx = stem.split("_", 1)[1] if "_" in stem else stem
        out_dir = tactile_root / f"snapshot_{idx}"

        log.info("Stage 7: rendering %s → %s", snap_path.name, out_dir)
        try:
            tactile_mod.run(snap_path, out_dir)
            successful.append(out_dir)
        except Exception as exc:
            log.error(
                "Stage 7: tactile rendering failed for %s: %s — continuing",
                snap_path.name, exc,
            )

    log.info(
        "Stage 7: %d/%d snapshots rendered successfully",
        len(successful), len(snapshot_paths),
    )
    return successful
