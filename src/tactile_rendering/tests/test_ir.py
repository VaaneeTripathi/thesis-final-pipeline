import json
from pathlib import Path
from jsonschema import validate, Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "static-schema.json"
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "v1_simple_flowchart.json"


def test_fixture_validates_against_schema():
    schema = json.loads(SCHEMA_PATH.read_text())
    instance = json.loads(FIXTURE_PATH.read_text())
    validate(instance=instance, schema=schema, cls=Draft202012Validator)


def test_fixture_has_expected_structure():
    instance = json.loads(FIXTURE_PATH.read_text())
    assert len(instance["elements"]["nodes"]) == 5
    assert len(instance["elements"]["connections"]) == 4
    assert instance["elements"]["nodes"][0]["id"] == "n1"
    assert instance["provenance"]["confidence"] == "high"
