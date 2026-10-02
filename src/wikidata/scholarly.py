"""Scholarly works: the entities a release's tables put in a set of their own.

philippesaade/wikidata left out "scholarly articles"; scripts/p31_survey.py found which
classes its copy of 2026-05-07 was missing (docs/journal/2026-10-01-official-dump-releases.md).
An entity is scholarly if any of its "instance of" (P31) value snaks is one of
SCHOLARLY_CLASSES: scholarly classes nearly all of whose entities that copy left out (at
least 99%), and six partly left out whose items are scholarly works too. `route-release` moves them from a release's chunks to the
scholarly set (config.SCHOLAR), published as the `wikidata-scholar-*` tables.
"""

import polars as pl

SCHOLARLY_CLASSES = {
    # At least 99% of their entities missing from philippesaade/wikidata
    "Q13442814": "scholarly article",
    "Q871232": "editorial",
    "Q2782326": "case report",
    "Q815382": "meta-analysis",
    "Q187685": "doctoral thesis",
    "Q1348305": "erratum",
    "Q1907875": "master's thesis",
    "Q18918145": "academic journal article",
    "Q45182324": "retracted paper",
    "Q23927052": "conference paper",
    "Q1402850": "field study report",
    "Q15781350": "final project report",
    "Q1266946": "thesis",
    "Q7316896": "retraction notice",
    "Q580922": "preprint",
    "Q5246046": "academic publishing",
    "Q30749496": "diploma thesis",
    "Q111475835": "bachelor's with honors thesis",
    "Q56478376": "expression of concern",
    "Q798134": "bachelor's thesis",
    "Q114613919": "scientific note",
    "Q92998777": "opinion paper",
    "Q58901591": "comparative study",
    "Q10885494": "academic conference paper",
    "Q132115645": "standard analytical method",
    "Q59387148": "research report",
    "Q1385450": "dissertation",
    "Q130709863": "book or chapter",
    "Q54670950": "conference poster",
    "Q51282918": "Doctor of Philosophy thesis",
    "Q51282711": "Doctor of Clinical Psychology thesis",
    "Q15706459": "research article",
    "Q111475860": "postgraduate diploma thesis",
    "Q51283092": "Master of Arts thesis",
    "Q58900768": "consensus statement",
    "Q58897583": "comment",
    "Q58898636": "evaluation study",
    # Partly missing (share of their entities missing)
    "Q591041": "scientific publication",  # 93%
    "Q193842": "geological map",  # 95%
    "Q58632367": "scholarly conference abstract",  # 89%
    "Q3099732": "technical report",  # 73%
    "Q637866": "book review",  # 44%
    "Q21481766": "scholarly chapter",  # 23%
}

# A P31 value snak in a chunk's claims JSON: anchored on `mainsnak` so a qualifier's P31
# does not count, and with no brace before its `datavalue` (which dump.py writes last), so
# it stays within the main snak whatever its other keys
P31_SNAK = (
    r'"mainsnak":\{[^{}]*"property":"P31"[^{}]*'
    r'"datavalue":\{"entity-type":"item","numeric-id":\d+,"id":"(Q\d+)"'
)


def p31(claims: pl.Expr) -> pl.Expr:
    """The P31 value ids in each row's claims JSON."""
    return claims.str.extract_all(P31_SNAK).list.eval(pl.element().str.extract(r'"id":"(Q\d+)"$'))


def is_scholarly(claims: pl.Expr) -> pl.Expr:
    """Whether each row's claims JSON has a P31 in SCHOLARLY_CLASSES."""
    return p31(claims).list.eval(pl.element().is_in(list(SCHOLARLY_CLASSES))).list.any()
