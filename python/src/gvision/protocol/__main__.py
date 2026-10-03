"""Write the JSON Schema for the protocol: ``python -m gvision.protocol [path]``."""

import json
import sys
from pathlib import Path

from gvision.protocol.messages import json_schema


def render() -> str:
    return json.dumps(json_schema(), indent=2) + "\n"


def main() -> None:
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(render(), encoding="utf-8")
    else:
        sys.stdout.write(render())


if __name__ == "__main__":
    main()
