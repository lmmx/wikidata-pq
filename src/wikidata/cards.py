"""Dataset cards: each table's README.md, rendered from its template in DATASET_CARDS_DIR
and the partition metadata in DATASET_CARDS_METADATA.

A template holds two placeholder lines: `{{configs}}` in the front matter becomes one
config (subset) per key directory, plus `all`, and `{{sizes}}` in the body becomes the
size of each key. The rendered cards are written to RENDERED_CARDS_DIR, and pushed to
the Hub as the last stage of `finalise`, only where they differ from the repo's README.md.
"""

from __future__ import annotations

import json

from huggingface_hub import HfApi

from .config import (
    DATASET_CARDS_DIR,
    DATASET_CARDS_METADATA,
    RENDERED_CARDS_DIR,
    Table,
)

# The subset each card loads by default: the largest key of every table split by language
DEFAULT_CONFIG = {Table.LINKS: "enwiki", Table.CLAIMS: "all"}
DEFAULT_CONFIG_FALLBACK = "en"
# Keys listed in the card's main size table, largest first; every key is in a <details>
INLINE_KEYS = 10
KEY_HEADING = {Table.LINKS: "Site"}


def read_metadata() -> dict[str, dict[str, dict[str, int]]]:
    path = DATASET_CARDS_METADATA
    return json.loads(path.read_text()) if path.exists() else {}


def _default_config(table: Table) -> str:
    return DEFAULT_CONFIG.get(table, DEFAULT_CONFIG_FALLBACK)


def _configs(table: Table, keys: list[str]) -> str:
    """YAML for the front matter's `configs`: one config per key directory, and `all`
    for every key. Names are quoted: a bare `no` (Norwegian) would load as false."""
    default = _default_config(table)
    configs = [(k, f"{k}/*.parquet") for k in keys if k != "all"]
    configs.append(("all", "*/*.parquet"))
    if default not in dict(configs):
        default = "all"
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
    by_size = sorted(keys.items(), key=lambda kv: (-kv[1]["bytes"], kv[0]))
    subsets = "sites" if table is Table.LINKS else "languages"
    return "\n\n".join(
        [
            f"In total: {len(keys)} {subsets}, {total}. The largest:",
            _size_table(table, by_size[:INLINE_KEYS]),
            f"<details>\n<summary>All {len(keys)} {subsets}</summary>",
            _size_table(table, by_size),
            "</details>",
        ]
    )


def render_card(table: Table, metadata: dict | None = None) -> str:
    """The table's card: its template with the configs and sizes of its keys. A table
    not in the metadata gets only the `all` config and no sizes."""
    metadata = read_metadata() if metadata is None else metadata
    keys = metadata.get(str(table), {})
    template = (DATASET_CARDS_DIR / f"{table}.md").read_text()
    for placeholder in ["{{configs}}", "{{sizes}}"]:
        if template.count(placeholder) != 1:
            raise ValueError(f"[cards] {table}: template needs one {placeholder}")
    card = template.replace("{{configs}}", _configs(table, sorted(keys)))
    card = card.replace("{{sizes}}", _sizes(table, keys) if keys else "")
    return card


def write_cards() -> dict[Table, str]:
    """Render every table's card to RENDERED_CARDS_DIR/{table}.md."""
    metadata = read_metadata()
    RENDERED_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    cards = {}
    for table in Table:
        cards[table] = render_card(table, metadata)
        (RENDERED_CARDS_DIR / f"{table}.md").write_text(cards[table])
    print(f"[cards] Rendered {len(cards)} cards to {RENDERED_CARDS_DIR}", flush=True)
    return cards


def _hub_card(repo_id: str, api: HfApi) -> str | None:
    if not api.file_exists(repo_id, "README.md", repo_type="dataset"):
        return None
    path = api.hf_hub_download(repo_id, "README.md", repo_type="dataset")
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
        commit_message="Update dataset card",
    )
    print(f"[cards] {repo_id}: card updated", flush=True)
    return True
