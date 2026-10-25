"""claims.avsc against the target schema (claims_schema) the published claims are in:

- every field of claims_schema is in claims.avsc, so a claim holding it is built by the
  split (claims_build) rather than sending its part to genson's inference; but for the
  promoted integer fields beside a float one (`latitude__integer`, `longitude__integer`),
  which are listed: genson's strict decode gives an integer to `__number` where its record
  has no integer field, as the published claims hold it (see avro_schemas.complete);
- each field claims.avsc has is of its target type (claims_build._plan raises otherwise);
- claims.avsc is its own completion: `scripts/avro_schemas.py --complete` leaves it as it
  is.

    WIKIDATA_RELEASE=20261012 .venv/bin/python scripts/test_claims_schema.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from avro_schemas import complete  # noqa: E402
from wikidata import claims_build  # noqa: E402
from wikidata.claims_schema import CLAIMS_AVRO, CLAIMS_AVRO_PATH  # noqa: E402

# The target fields claims.avsc leaves out, each a promoted integer beside its float field
SNAKS = (".claims{}[].mainsnak", ".claims{}[].qualifiers{}[]", ".claims{}[].references[].snaks{}[]")
LEFT_OUT = {f"{snak}.datavalue.{c}.{c}__integer" for snak in SNAKS for c in ("latitude", "longitude")}


def phantoms(plan: tuple, path: str = "") -> set[str]:
    """The paths of the target fields a decode plan has no Avro type for."""
    kind = plan[0]
    if kind == "phantom":
        return {path}
    if kind == "record":
        return {p for name, child in plan[2] for p in phantoms(child, f"{path}.{name}")}
    if kind == "map":
        return phantoms(plan[2], f"{path}{{}}")
    if kind == "array":
        return phantoms(plan[2], f"{path}[]")
    return set()


def main() -> None:
    assert CLAIMS_AVRO is not None and claims_build.PLAN is not None, "no claims.avsc, or the build is off"
    missing = phantoms(claims_build.PLAN, ".claims")
    assert missing == LEFT_OUT, (
        f"target fields claims.avsc lacks: {sorted(missing - LEFT_OUT)}; "
        f"listed but present: {sorted(LEFT_OUT - missing)} (rerun scripts/avro_schemas.py --complete)"
    )
    print(f"claims.avsc: every field of claims_schema, but the {len(LEFT_OUT)} promoted integers listed")
    assert complete(CLAIMS_AVRO) + "\n" == CLAIMS_AVRO, f"{CLAIMS_AVRO_PATH} is not complete: run avro_schemas.py --complete"
    print("claims.avsc: its own completion")


if __name__ == "__main__":
    main()
