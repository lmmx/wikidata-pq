"""Dataset cards: each table's README.md, rendered from its template in CARD_TEMPLATES_DIR,
the partition metadata in DATASET_CARDS_METADATA and the figures in DATASET_CARDS_STATS.

Every statement about the data in a card comes from those two files, through the
placeholders below; the templates hold only the text that does not depend on the data.
Rendering fails on figures not computed from the current metadata and card_stats spec, on a
placeholder left unfilled, and on an example that names a subset the repo does not have.

- `{{configs}}` (front matter): one config (subset) per key directory, plus `all`.
- `{{default}}`: the default config.
- `{{key_examples}}`: the largest keys, and the largest one with a hyphen.
- `{{sample}}`: the rows of a few fixed ids (card_stats.SAMPLES).
- `{{sizes}}`: totals, the largest keys and every key.
- `{{languages}}`: coverage of `en` and `mul` (card_stats.STATS_COLUMN tables).

The rendered cards are written to RENDERED_CARDS_DIR, and pushed to the Hub as the last
stage of `finalise`, only where they differ from the repo's README.md.
"""

from __future__ import annotations

import json
import re

from huggingface_hub import HfApi

from .card_stats import SAMPLES, STATS_COLUMN, current, read_metadata, read_stats
from .config import (
    HUB_REVISION,
    CARD_TEMPLATES_DIR,
    RELEASE,
    DATASET_CARDS_STATS,
    HF_USER,
    RENDERED_CARDS_DIR,
    Table,
)

# The subset each card loads by default: the largest key of every table split by language
DEFAULT_CONFIG = {Table.LINKS: "enwiki", Table.CLAIMS: "all"}
DEFAULT_CONFIG_FALLBACK = "en"
# Keys listed in the card's main size table, largest first; every key is in a <details>
INLINE_KEYS = 10
KEY_HEADING = {Table.LINKS: "Site"}
KEY_NOUN = {Table.LINKS: "sites"}
KEY_NOUN_FALLBACK = "languages"

PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
# A subset named in an example: a path in a repo, or a config passed to load_dataset
EXAMPLE_PATH = re.compile(r"wikidata-(?:scholar-)?(\w+)/([^/{}*\"\s]+)/\*")
EXAMPLE_CONFIG = re.compile(rf'load_dataset\("{HF_USER}/wikidata-(?:scholar-)?(\w+)", "([^"]+)"')

MUL_HELP = "https://www.wikidata.org/wiki/Help:Default_values_for_labels_and_aliases"
COVERAGE_TEXT = {
    Table.LABEL: "Not every item has a label in every language: of the {any} items with "
    "a label, {en} ({en_pct}) have one in `en`.",
    Table.DESC: "Of the {any} items with a description, {en} ({en_pct}) have one in "
    "`en`.",
    Table.ALIAS: "Of the {any} items with an alias, {en} ({en_pct}) have one in `en`.",
    Table.CLAIMS_LABELS: "Of the {any} items that statements refer to (as values or "
    "units), {en} ({en_pct}) have a name in `en`.",
}
MUL_TEXT = {
    Table.LABEL: "`mul` is Wikidata's code for a [default label]({help}), one that "
    "holds in every language, such as a person's name in the Latin alphabet. {mul} items "
    "have a `mul` label, and {mul_no_en} of them ({mul_no_en_pct} of items with a label) "
    "have no `en` label, so reading `en` alone misses their names.",
    Table.ALIAS: "`mul` is Wikidata's code for [default aliases]({help}), ones that hold "
    "in every language. {mul} items have `mul` aliases, and {mul_no_en} of them "
    "({mul_no_en_pct} of items with an alias) have no `en` alias, so reading `en` alone "
    "misses those.",
    Table.CLAIMS_LABELS: "`mul` is Wikidata's code for a [default label]({help}), one "
    "that holds in every language. {mul} of these items have a `mul` label, and "
    "{mul_no_en} of them ({mul_no_en_pct}) have no `en` label, so reading `en` alone "
    "leaves them unnamed.",
}
NO_MUL_TEXT = (
    "There is no `mul` subset: Wikidata's [default values]({help}), in the `mul` code, "
    "are for labels and aliases only."
)


def _default_config(table: Table, keys: list[str]) -> str:
    default = DEFAULT_CONFIG.get(table, DEFAULT_CONFIG_FALLBACK)
    return default if default in keys else "all"


def _configs(table: Table, keys: list[str]) -> str:
    """YAML for the front matter's `configs`: one config per key directory, and `all`
    for every key. Names are quoted: a bare `no` (Norwegian) would load as false."""
    default = _default_config(table, keys)
    configs = [(k, f"{k}/*.parquet") for k in keys if k != "all"]
    configs.append(("all", "*/*.parquet"))
    lines = ["configs:"]
    for name, data_files in configs:
        lines.append(f'- config_name: "{name}"')
        lines.append(f'  data_files: "{data_files}"')
        if name == default:
            lines.append("  default: true")
    return "\n".join(lines)


def _size(n: float) -> str:
    """Bytes in decimal units, as the Hub shows them."""
    if n < 1000:
        return f"{n:.0f} B"
    for unit in ["kB", "MB"]:
        n /= 1000
        if n < 1000:
            return f"{n:.1f} {unit}"
    return f"{n / 1000:.1f} GB"


def _by_size(keys: dict[str, dict[str, int]]) -> list[tuple[str, dict[str, int]]]:
    return sorted(keys.items(), key=lambda kv: (-kv[1]["bytes"], kv[0]))


def _key_examples(keys: dict[str, dict[str, int]]) -> str:
    """The three largest keys and the largest with a hyphen, e.g. `en`, `nl`, ..."""
    names = [k for k, _ in _by_size(keys)]
    examples = names[:3] + [k for k in names[3:] if "-" in k][:1]
    return ", ".join(f"`{k}`" for k in examples) + ", ..."


def _size_table(table: Table, rows: list[tuple[str, dict[str, int]]]) -> str:
    heading = KEY_HEADING.get(table, "Language")
    lines = [
        f"| {heading} | Files | Size | Rows |",
        "|---|--:|--:|--:|",
    ]
    for key, m in rows:
        lines.append(
            f"| `{key}` | {m['files']} | {_size(m['bytes'])} | {m['rows']:,} |"
        )
    return "\n".join(lines)


def _sizes(table: Table, keys: dict[str, dict[str, int]]) -> str:
    """The card's size section: totals, the largest keys and every key."""
    files = sum(m["files"] for m in keys.values())
    size = sum(m["bytes"] for m in keys.values())
    rows = sum(m["rows"] for m in keys.values())
    total = f"{files} files, {_size(size)} of Parquet, {rows:,} rows"
    if len(keys) == 1:
        return f"In total: {total}."
    by_size = _by_size(keys)
    noun = KEY_NOUN.get(table, KEY_NOUN_FALLBACK)
    return "\n\n".join(
        [
            f"In total: {len(keys)} {noun}, {total}. The largest:",
            _size_table(table, by_size[:INLINE_KEYS]),
            f"<details>\n<summary>All {len(keys)} {noun}</summary>",
            _size_table(table, by_size),
            "</details>",
        ]
    )


def _cell(v: object) -> str:
    """A sample value as shown: strings as they are, others (null, a list such as a
    sitelink's badges) as JSON."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _sample(rows: list[dict[str, object]]) -> str:
    """Rows as a fixed-width block, one column per field."""
    cols = list(rows[0])
    lines = [[c for c in cols]] + [[_cell(r[c]) for c in cols] for r in rows]
    widths = [max(len(line[i]) for line in lines) for i in range(len(cols))]
    body = "\n".join(
        "  ".join(v.ljust(w) for v, w in zip(line, widths)).rstrip() for line in lines
    )
    return f"```\n{body}\n```"


def _languages(table: Table, keys: dict[str, dict[str, int]], entry: dict) -> str:
    """Coverage of `en` and `mul` among the table's items (`Q` ids)."""
    if "en" not in keys:
        raise ValueError(f"[cards] {table}: no `en` key to give coverage of")
    q = entry["coverage"]["Q"]
    figures = {k: f"{v:,}" for k, v in q.items()}
    figures |= {f"{k}_pct": f"{100 * v / q['any']:.1f}%" for k, v in q.items()}
    paragraphs = [COVERAGE_TEXT[table].format(**figures)]
    if "mul" in keys:
        if table not in MUL_TEXT:
            raise ValueError(f"[cards] {table}: a `mul` key, and no text for it")
        paragraphs.append(MUL_TEXT[table].format(help=MUL_HELP, **figures))
    else:
        paragraphs.append(NO_MUL_TEXT.format(help=MUL_HELP))
    return "\n\n".join(paragraphs)


def _check_examples(table: Table, card: str, metadata: dict) -> None:
    """Refuse an example naming a subset that its repo does not have."""
    named = [(t, k, False) for t, k in EXAMPLE_PATH.findall(card)]
    named += [(t, k, True) for t, k in EXAMPLE_CONFIG.findall(card)]
    for t, key, is_config in named:
        if t not in metadata:
            continue
        if key not in metadata[t] and not (is_config and key == "all"):
            raise ValueError(f"[cards] {table}: example names {t}/{key}, not a subset")


def render_card(
    table: Table, metadata: dict | None = None, stats: dict | None = None
) -> str:
    """The table's card, from its template, the metadata and the figures. A table not
    in the metadata (a repo's first push) gets only the `all` config, and every other
    placeholder left empty."""
    metadata = read_metadata() if metadata is None else metadata
    stats = read_stats() if stats is None else stats
    keys = metadata.get(str(table), {})
    template = (CARD_TEMPLATES_DIR / f"{table}.md").read_text()
    needs_stats = bool(keys) and table in SAMPLES
    if needs_stats and not current(table, metadata, stats):
        raise ValueError(
            f"[cards] {table}: {DATASET_CARDS_STATS.name} is not computed from the "
            "current metadata and card_stats spec: run card-stats"
        )
    entry = stats.get(str(table), {})
    values = {"configs": _configs(table, sorted(keys))}
    if RELEASE:
        values["release"] = RELEASE
    if keys:
        values["default"] = f"`{_default_config(table, sorted(keys))}`"
        values["key_examples"] = _key_examples(keys)
        values["sizes"] = _sizes(table, keys)
    if needs_stats:
        values["sample"] = _sample(entry["sample"])
    if needs_stats and table in STATS_COLUMN:
        values["languages"] = _languages(table, keys, entry)
    found = PLACEHOLDER.findall(template)
    if found.count("configs") != 1:
        raise ValueError(f"[cards] {table}: template needs one {{{{configs}}}}")
    unknown = sorted(
        set(found)
        - {"configs", "default", "key_examples", "sample", "sizes", "languages", "release"}
    )
    if unknown:
        raise ValueError(f"[cards] {table}: unknown placeholders {unknown}")
    if keys:
        unfilled = sorted(set(found) - set(values))
        if unfilled:
            raise ValueError(f"[cards] {table}: nothing to fill {unfilled} with")
    card = PLACEHOLDER.sub(lambda m: values.get(m.group(1), ""), template)
    _check_examples(table, card, metadata)
    return card


def write_cards() -> dict[Table, str]:
    """Render every table's card to RENDERED_CARDS_DIR/{table}.md."""
    metadata, stats = read_metadata(), read_stats()
    cards = {table: render_card(table, metadata, stats) for table in Table}
    RENDERED_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    for table, card in cards.items():
        (RENDERED_CARDS_DIR / f"{table}.md").write_text(card)
    print(f"[cards] Rendered {len(cards)} cards to {RENDERED_CARDS_DIR}", flush=True)
    return cards


def _hub_card(repo_id: str, api: HfApi) -> str | None:
    if not api.file_exists(repo_id, "README.md", repo_type="dataset", revision=HUB_REVISION):
        return None
    path = api.hf_hub_download(repo_id, "README.md", repo_type="dataset", revision=HUB_REVISION)
    with open(path) as f:
        return f.read()


def push_card(repo_id: str, card: str, api: HfApi | None = None) -> bool:
    """Upload the card as the repo's README.md if it differs from the one there."""
    api = api or HfApi()
    if _hub_card(repo_id, api) == card:
        print(f"[cards] {repo_id}: card up to date", flush=True)
        return False
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        revision=HUB_REVISION,
        commit_message="Update dataset card",
    )
    print(f"[cards] {repo_id}: card updated", flush=True)
    return True
