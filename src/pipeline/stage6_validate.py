"""Stage 6: Schema validation and semantic delta computation.

Responsibilities:
  1. Validate all static IR snapshots against static-schema.json
  2. Validate the aggregated operation document against operation-schema.json
  3. Verify operation summary counts match the operations array
  4. Compute semantic graph deltas between consecutive static snapshots
     (added/removed/modified nodes and connections)
  5. Verify Mealy state sequence has no undefined transitions
  6. Round-trip check via RFC 6902 patch (internal — not written to disk)

Outputs written to output_dir/:
  snapshots/snapshot_NNNN.json   — validated static IR files
  operations/operations.json     — single validated operation document
  deltas/delta_NNNN.json         — semantic delta between snapshot N and N+1
  validation_report.json         — collected errors
"""
from __future__ import annotations
import copy
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


@dataclass
class ValidationReport:
    static_errors: list[str] = field(default_factory=list)
    operation_errors: list[str] = field(default_factory=list)
    round_trip_errors: list[str] = field(default_factory=list)   # internal only
    mealy_errors: list[str] = field(default_factory=list)
    summary_count_errors: list[str] = field(default_factory=list)

    @property
    def all_errors(self) -> list[str]:
        return (
            self.static_errors
            + self.operation_errors
            + self.round_trip_errors
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


def _load_schema(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_static_ir(snapshot: dict, validator: Draft202012Validator) -> list[str]:
    return [
        f"static-schema: {err.json_path}: {err.message}"
        for err in validator.iter_errors(snapshot)
    ]


def validate_operation_ir(op_doc: dict, validator: Draft202012Validator) -> list[str]:
    return [
        f"operation-schema: {err.json_path}: {err.message}"
        for err in validator.iter_errors(op_doc)
    ]


def validate_summary_counts(op_doc: dict) -> list[str]:
    """Verify operation_type counts in summary match the operations array."""
    errors = []
    try:
        ops = op_doc["analysis"]["operations"]
        summary = op_doc["analysis"]["summary"]
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


def validate_mealy_history(history: list[dict]) -> list[str]:
    return [
        f"undefined Mealy transition at step {i}: "
        f"({entry['from_state']}, {entry['symbol']}) — context: {entry.get('context')}"
        for i, entry in enumerate(history)
        if entry.get("error") == "undefined_transition"
    ]

def _compute_patch(snapshot_a: dict, snapshot_b: dict) -> list[dict]:
    return jsonpatch.make_patch(snapshot_a, snapshot_b).patch


def _round_trip_check(
    snapshot_a: dict,
    patch: list[dict],
    snapshot_b: dict,
) -> list[str]:
    """Verify applying patch to snapshot_a reproduces snapshot_b exactly."""
    errors = []
    try:
        result = jsonpatch.apply_patch(copy.deepcopy(snapshot_a), patch)
        if result != snapshot_b:
            errors.append(
                f"round-trip check failed for snapshot id={snapshot_b.get('id', '?')}: "
                "patch applied to N does not reproduce N+1"
            )
    except Exception as exc:
        errors.append(f"round-trip check exception: {exc}")
    return errors


def compute_semantic_delta(snapshot_a: dict, snapshot_b: dict) -> dict:
    """Compute a semantic graph delta between two consecutive static IR snapshots.

    Nodes are matched by id. Connections are matched by (source, target, direction).
    Modification = same key in both snapshots but any field differs.

    Returns:
        {
          "from_snapshot": str,     — id of snapshot_a
          "to_snapshot":   str,     — id of snapshot_b
          "added_nodes":   [...],   — nodes in B but not A
          "removed_nodes": [...],   — nodes in A but not B
          "modified_nodes": [{"before": {...}, "after": {...}}],
          "added_connections":   [...],
          "removed_connections": [...]
        }
    """
    nodes_a = {
        n["id"]: n
        for n in snapshot_a.get("elements", {}).get("nodes", [])
    }
    nodes_b = {
        n["id"]: n
        for n in snapshot_b.get("elements", {}).get("nodes", [])
    }

    ids_a = set(nodes_a)
    ids_b = set(nodes_b)

    added_nodes = [nodes_b[nid] for nid in sorted(ids_b - ids_a)]
    removed_nodes = [nodes_a[nid] for nid in sorted(ids_a - ids_b)]
    modified_nodes = [
        {"before": nodes_a[nid], "after": nodes_b[nid]}
        for nid in sorted(ids_a & ids_b)
        if nodes_a[nid] != nodes_b[nid]
    ]

    def _conn_key(c: dict) -> tuple:
        return (c.get("source"), c.get("target"), c.get("direction", "forward"))

    conns_a = {_conn_key(c): c for c in snapshot_a.get("elements", {}).get("connections", [])}
    conns_b = {_conn_key(c): c for c in snapshot_b.get("elements", {}).get("connections", [])}

    keys_a = set(conns_a)
    keys_b = set(conns_b)

    added_connections = [conns_b[k] for k in keys_b - keys_a]
    removed_connections = [conns_a[k] for k in keys_a - keys_b]

    return {
        "from_snapshot": snapshot_a.get("id"),
        "to_snapshot": snapshot_b.get("id"),
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "modified_nodes": modified_nodes,
        "added_connections": added_connections,
        "removed_connections": removed_connections,
    }


def run(
    snapshots: list[dict],
    operation_doc: dict,
    mealy_history: list[dict],
    output_dir: Path,
) -> tuple[ValidationReport, list[dict]]:
    """Validate all pipeline outputs, compute semantic deltas, write results.

    Args:
        snapshots:      All static IR dicts in chronological order.
        operation_doc:  Single aggregated operation IR document for the whole video.
        mealy_history:  MealyMachine.history after the full run.
        output_dir:     Root output directory.

    Returns:
        (ValidationReport, deltas) where deltas[i] is the semantic delta
        from snapshots[i] to snapshots[i+1].
    """
    report = ValidationReport()

    static_schema = _load_schema(_STATIC_SCHEMA_PATH)
    op_schema = _load_schema(_OPERATION_SCHEMA_PATH)
    static_validator = Draft202012Validator(static_schema)
    op_validator = Draft202012Validator(op_schema)

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    operations_dir = output_dir / "operations"
    deltas_dir = output_dir / "deltas"
    snapshots_dir.mkdir(exist_ok=True)
    operations_dir.mkdir(exist_ok=True)
    deltas_dir.mkdir(exist_ok=True)

    #  Static IR: validate + write 
    for i, snapshot in enumerate(snapshots):
        errs = validate_static_ir(snapshot, static_validator)
        if errs:
            report.static_errors.extend(
                f"snapshot[{i}] (id={snapshot.get('id', '?')}): {e}" for e in errs
            )
        (snapshots_dir / f"snapshot_{i:04d}.json").write_text(
            json.dumps(snapshot, indent=2)
        )

    log.info(
        "Stage 6: validated %d static snapshots (%d errors)",
        len(snapshots), len(report.static_errors),
    )

    # Operation IR: validate + write (single document)
    errs = validate_operation_ir(operation_doc, op_validator)
    report.operation_errors.extend(errs)
    report.summary_count_errors.extend(validate_summary_counts(operation_doc))
    (operations_dir / "operations.json").write_text(json.dumps(operation_doc, indent=2))

    log.info(
        "Stage 6: validated operation document (%d schema errors, %d count errors)",
        len(report.operation_errors), len(report.summary_count_errors),
    )

    # Semantic deltas: compute + write 
    deltas: list[dict] = []
    for i in range(len(snapshots) - 1):
        delta = compute_semantic_delta(snapshots[i], snapshots[i + 1])
        deltas.append(delta)
        (deltas_dir / f"delta_{i:04d}.json").write_text(json.dumps(delta, indent=2))

        # Internal round-trip check via RFC 6902 patch (not written to disk)
        patch = _compute_patch(snapshots[i], snapshots[i + 1])
        report.round_trip_errors.extend(
            _round_trip_check(snapshots[i], patch, snapshots[i + 1])
        )

    log.info(
        "Stage 6: computed %d semantic deltas (%d round-trip errors)",
        len(deltas), len(report.round_trip_errors),
    )

    # Mealy history validation 
    report.mealy_errors.extend(validate_mealy_history(mealy_history))
    if report.mealy_errors:
        log.warning("Stage 6: %d undefined Mealy transition(s)", len(report.mealy_errors))

    # Validation report 
    (output_dir / "validation_report.json").write_text(json.dumps({
        "is_valid": report.is_valid,
        "static_errors": report.static_errors,
        "operation_errors": report.operation_errors,
        "round_trip_errors": report.round_trip_errors,
        "mealy_errors": report.mealy_errors,
        "summary_count_errors": report.summary_count_errors,
    }, indent=2))

    report.log_summary()
    return report, deltas
