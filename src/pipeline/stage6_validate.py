"""Stage 6: JSON Patch computation and schema validation.

Responsibilities:
  1. Validate all static IR snapshots against static-schema.json
  2. Validate all operation IRs against operation-schema.json
  3. Compute RFC 6902 JSON Patches between consecutive static snapshots
  4. Round-trip check: applying patch N to snapshot N reproduces snapshot N+1
  5. Verify Mealy state sequence has no undefined transitions
  6. Verify operation-schema summary counts match the operations array

All issues are collected and returned; nothing is raised unless the caller
asks for strict mode.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import jsonpatch
import jsonschema
from jsonschema import Draft202012Validator

log = logging.getLogger(__name__)

_SCHEMAS_DIR = Path(__file__).parent.parent.parent / "schemas"
_STATIC_SCHEMA_PATH = _SCHEMAS_DIR / "static-schema.json"
_OPERATION_SCHEMA_PATH = _SCHEMAS_DIR / "operation-schema.json"


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    static_errors: list[str] = field(default_factory=list)
    operation_errors: list[str] = field(default_factory=list)
    patch_errors: list[str] = field(default_factory=list)
    mealy_errors: list[str] = field(default_factory=list)
    summary_count_errors: list[str] = field(default_factory=list)

    @property
    def all_errors(self) -> list[str]:
        return (
            self.static_errors
            + self.operation_errors
            + self.patch_errors
            + self.mealy_errors
            + self.summary_count_errors
        )

    @property
    def is_valid(self) -> bool:
        return len(self.all_errors) == 0

    def log_summary(self) -> None:
        if self.is_valid:
            log.info("Stage 6: all validations passed.")
        else:
            log.warning("Stage 6: %d validation error(s):", len(self.all_errors))
            for err in self.all_errors:
                log.warning("  - %s", err)


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def _load_schema(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def validate_static_ir(snapshot: dict, validator: Draft202012Validator) -> list[str]:
    errors = []
    for err in validator.iter_errors(snapshot):
        errors.append(f"static-schema: {err.json_path}: {err.message}")
    return errors


def validate_operation_ir(op_ir: dict, validator: Draft202012Validator) -> list[str]:
    errors = []
    for err in validator.iter_errors(op_ir):
        errors.append(f"operation-schema: {err.json_path}: {err.message}")
    return errors


def validate_summary_counts(op_ir: dict) -> list[str]:
    """Verify that operation_type counts in summary match the operations array."""
    errors = []
    try:
        ops = op_ir["analysis"]["operations"]
        summary = op_ir["analysis"]["summary"]
        type_counts: dict[str, int] = {}
        for op in ops:
            t = op["operation_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        mapping = {
            "CREATION": "creation_count",
            "ADDITION": "addition_count",
            "HIGHLIGHTING": "highlighting_count",
            "ERASURE": "erasure_count",
            "COMPLETE_ERASURE": "complete_erasure_count",
        }
        for op_type, field_name in mapping.items():
            expected = type_counts.get(op_type, 0)
            actual = summary.get(field_name, 0)
            if expected != actual:
                errors.append(
                    f"summary count mismatch: {field_name}={actual} but "
                    f"{op_type} appears {expected} time(s) in operations array"
                )
    except (KeyError, TypeError) as exc:
        errors.append(f"summary count validation error: {exc}")
    return errors


def compute_patch(snapshot_a: dict, snapshot_b: dict) -> list[dict]:
    """Compute RFC 6902 JSON Patch from snapshot_a to snapshot_b."""
    return jsonpatch.make_patch(snapshot_a, snapshot_b).patch


def round_trip_check(
    snapshot_a: dict,
    patch: list[dict],
    snapshot_b: dict,
) -> list[str]:
    """Verify applying patch to snapshot_a reproduces snapshot_b."""
    errors = []
    try:
        result = jsonpatch.apply_patch(snapshot_a, patch)
        if result != snapshot_b:
            errors.append(
                f"round-trip check failed: patch applied to snapshot N does not "
                f"reproduce snapshot N+1 (id={snapshot_b.get('id', '?')})"
            )
    except Exception as exc:
        errors.append(f"round-trip check exception: {exc}")
    return errors


def validate_mealy_history(history: list[dict]) -> list[str]:
    """Check that the Mealy machine history contains no undefined transitions."""
    return [
        f"undefined Mealy transition at step {i}: "
        f"({entry['from_state']}, {entry['symbol']}) — context: {entry.get('context')}"
        for i, entry in enumerate(history)
        if entry.get("error") == "undefined_transition"
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    snapshots: list[dict],
    operation_irs: list[dict],
    patches: list[list[dict]],
    mealy_history: list[dict],
    output_dir: Path,
) -> ValidationReport:
    """Validate all pipeline outputs and write them to output_dir.

    Args:
        snapshots:       All static IR dicts (chronological order).
        operation_irs:   All operation IR dicts (one per Mealy P emission).
        patches:         Patches[i] transforms snapshots[i] -> snapshots[i+1].
        mealy_history:   MealyMachine.history after the full run.
        output_dir:      Directory to write snapshots/, operations/, patches/.

    Returns:
        ValidationReport with all errors collected.
    """
    report = ValidationReport()

    static_schema = _load_schema(_STATIC_SCHEMA_PATH)
    op_schema = _load_schema(_OPERATION_SCHEMA_PATH)

    static_validator = Draft202012Validator(static_schema)
    op_validator = Draft202012Validator(op_schema)

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    operations_dir = output_dir / "operations"
    patches_dir = output_dir / "patches"
    snapshots_dir.mkdir(exist_ok=True)
    operations_dir.mkdir(exist_ok=True)
    patches_dir.mkdir(exist_ok=True)

    # --- Static IR validation + write ---
    for i, snapshot in enumerate(snapshots):
        errs = validate_static_ir(snapshot, static_validator)
        if errs:
            report.static_errors.extend(
                f"snapshot[{i}] (id={snapshot.get('id', '?')}): {e}" for e in errs
            )
        snap_path = snapshots_dir / f"snapshot_{i:04d}.json"
        snap_path.write_text(json.dumps(snapshot, indent=2))

    log.info("Stage 6: validated %d static snapshots (%d errors)", len(snapshots), len(report.static_errors))

    # --- Operation IR validation + write ---
    for i, op_ir in enumerate(operation_irs):
        errs = validate_operation_ir(op_ir, op_validator)
        if errs:
            report.operation_errors.extend(
                f"operation_ir[{i}]: {e}" for e in errs
            )
        count_errs = validate_summary_counts(op_ir)
        report.summary_count_errors.extend(count_errs)

        op_path = operations_dir / f"operation_{i:04d}.json"
        op_path.write_text(json.dumps(op_ir, indent=2))

    log.info("Stage 6: validated %d operation IRs (%d errors)", len(operation_irs), len(report.operation_errors))

    # --- Patch validation + write ---
    for i, patch in enumerate(patches):
        if i + 1 < len(snapshots):
            errs = round_trip_check(snapshots[i], patch, snapshots[i + 1])
            report.patch_errors.extend(errs)

        patch_path = patches_dir / f"patch_{i:04d}.json"
        patch_path.write_text(json.dumps(patch, indent=2))

    log.info("Stage 6: validated %d patches (%d errors)", len(patches), len(report.patch_errors))

    # --- Mealy history validation ---
    report.mealy_errors.extend(validate_mealy_history(mealy_history))
    if report.mealy_errors:
        log.warning("Stage 6: %d undefined Mealy transition(s)", len(report.mealy_errors))

    # --- Write validation report ---
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps({
        "is_valid": report.is_valid,
        "static_errors": report.static_errors,
        "operation_errors": report.operation_errors,
        "patch_errors": report.patch_errors,
        "mealy_errors": report.mealy_errors,
        "summary_count_errors": report.summary_count_errors,
    }, indent=2))

    report.log_summary()
    return report
