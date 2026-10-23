"""The harness's version (semver, from pyproject.toml). Results compare only within a major
version: a change to what is measured or how (the workload, the timing, parity) is a new
major version; a change to how results are shown is a minor or patch version.

Usage: version.py  (prints the version)
"""

import tomllib
from pathlib import Path

VERSION = tomllib.loads((Path(__file__).parent / "pyproject.toml").read_text())["project"]["version"]


def compatible(other: str | None, version: str = VERSION) -> bool:
    """Whether results of harness `other` compare with results of `version`."""
    return other is not None and other.split(".")[0] == version.split(".")[0]


if __name__ == "__main__":
    print(VERSION)
