import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gvision.protocol import ClearMsg, DimMsg, dump, parse
from gvision.protocol.__main__ import render
from gvision.demo import fake_objects

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"


def test_committed_schema_is_up_to_date():
    committed = (SCHEMA_DIR / "messages.schema.json").read_text(encoding="utf-8")
    assert committed == render(), "run: python -m gvision.protocol ../schema/messages.schema.json"


def test_examples_cover_every_message_type():
    examples = json.loads((SCHEMA_DIR / "examples.json").read_text(encoding="utf-8"))
    types = {parse(json.dumps(e)).type for e in examples}
    mapping = json.loads(render())["discriminator"]["mapping"]
    assert types == set(mapping)


def test_round_trip():
    msg = DimMsg(on=True, strength=0.4)
    assert parse(dump(msg)) == msg
    assert json.loads(dump(ClearMsg()))["v"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"v": 1, "ts": 0, "type": "nope"},
        {"v": 2, "ts": 0, "type": "clear", "reason": None},
        {"v": 1, "ts": 0, "type": "dim", "on": True, "strength": 1.5},
        {"v": 1, "ts": 0, "type": "focus", "refs": ["enemy"], "segment_id": None},
        {"v": 1, "ts": 0, "type": "clear", "reason": None, "extra": 1},
    ],
)
def test_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        parse(json.dumps(bad))


def test_demo_objects_are_valid():
    for obj in fake_objects(1.23):
        assert obj.status == "confirmed"
