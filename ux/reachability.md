# Reachability audit of the FAQs (ux/faqs.md) against the docs

Read: README, `docs/index`, `guide/*`, `reference/{config,hub,finalise,push,dump,cards,scripts,pull}`,
`changelog`, and the dataset card templates in `docs/dataset_cards/` (`dump/` for releases).
Not read in full: the other reference pages, `journal/`.

Scale for "hops" = pages/clicks from where a user would start (README for data users,
docs site for builders) to the sentence that answers it.

- **1** answered where you'd look first
- **2-3** answered, but you have to guess which page
- **4+ / inferred** only derivable by combining pages, or by reading the code
- **none** not in anything I read

## Structural problems (they explain most of the bad rows)

1. **Two audiences, one entry point, and the docs site only serves one of them.** The mkdocs
   site (`guide/`, `reference/`) is entirely for someone *running the pipeline*. Everything a
   *data consumer* needs (schema, `datavalue` fields, joins, `mul`, fallbacks, sort order, subsets,
   license) is in the dataset cards, and `mkdocs.yml` has `dataset_cards/` in `exclude_docs`.
   So the docs site cannot answer "how do I read a statement" at all; the answer is on the
   Hub card, which a GitHub-README reader is only told exists ("the card gives it").
2. **README is stale vs the docs.** README: "six Parquet datasets", 35.5 GB, built from the
   philippesaade copy, `just run` = "pull, process, partition and push". Docs index: seven
   tables, two sets (`wikidata-*` and `wikidata-scholar-*`), `main` holds the latest official-dump
   release. A user can't tell which one `main` is. Also README's `just run` vs `just finalise`
   split contradicts `source-copy.md` ("`run` calls `finalise` itself at the end").
3. **No "I just want to use the data" page and no troubleshooting/FAQ page.** Nothing in docs
   is organised by task. Everything is organised by module (`reference/`) or by stage
   (`guide/release`).
4. **Configuration that is source-edit-only is never labelled as such.** `HF_USER`,
   `HF_REPO_PRIVATE`, `CLEAN_UP_LOCAL`, `PREFETCH_*`, group sizes, hub dir: `config.md` lists
   five env vars and a constants table; a user has to infer that everything else means "edit
   `config.py`". Only `setup.md` says so, for `HF_REPO_PRIVATE` only.
5. **Two build modes share one set of command names**, and the docs describe the release mode
   as primary. Anything about disk, `hub/`, `just download`, `just run`, prefetch is in
   `source-copy.md` / `pull.md` / README and disagrees on where it lives.
6. **No glossary on the docs site** (Terminology is only in the README). `mul`, `snak`,
   `item/property/statement`, `row group` are used without definition in the cards and docs.
7. **Release vs main schema differences are scattered** (README, `source-copy.md`, two sets of
   cards) with no side-by-side.

## Verdict per FAQ

Numbers refer to `ux/faqs.md`.

### A. Which table do I want

| # | Verdict | Hops | Where / what's wrong |
|---|---|---|---|
| 1 label vs alias vs description | OK | 1 | README table, one line each. No example of the difference. |
| 2 facts about an item | OK | 1-2 | README example, then claims card "Names". |
| 3 readable names for `P31` | OK | 2 | claims card "Names" + claims_labels card. Good worked join (property only). |
| 4 `field` / `ref` meanings | Good | 1 | claims_labels card table. Docs site: none. |
| 5 Wikipedia page / sites | Weak | 2-3 | README terminology says "such as `enwiki`", links card has the key list. Never says `enwiki` ≠ `en`, or how to find Wikivoyage/Commons codes without opening the card. |
| 6 Q vs P, item type | **None** (main) | - | No `type` column in main. Only the release `entities` table has it, and nothing says "to find what kind of thing an id is, use entities". |
| 7 lexemes/senses/forms | **None** | - | `entity-type` mentions lexeme/form/sense in the release claims card, but nothing says whether lexemes are rows anywhere. |
| 8 entities table | **Contradictory** | 3 | README: six tables; "seventh for releases". Docs index: seven. Unclear which is on `main`. |
| 9 scholarly split | Weak | 2-3 | docs index + release cards. Cards say "these tables hold everything else". Never says "if you can't find a paper, it's in scholar". |
| 10 which repos are old vs new | **Contradictory** | 3 | README vs docs index (structural #2). Tag `20260507` explained in `hub.md` and cards only. |

### B. Loading

| # | Verdict | Hops | Where / what's wrong |
|---|---|---|---|
| 11 valid subset names | Weak | 2 | Card `{{sizes}}`/`{{languages}}` on Hub only. README: "the card gives it". No `list` snippet (the labels card does use `HfFileSystem().ls`, buried in the fallback code). |
| 12 what is `all` | Partial | 1 | Card: "`all` holds every language" (so yes stacked, but never says rows repeat per language, or that for claims `all` is the *only* subset and means something else). |
| 13 several languages with `load_dataset` | **None** | - | Only polars `concat` shown (README). |
| 14 slow / auth / rate limits on `hf://` | **None** | - | Nothing about tokens for reading, throttling, or when to download first. |
| 15 sizes/rows before downloading | OK | 2 | `{{sizes}}` in card, Hub only. |
| 16 streaming | Partial | 1 | Only on claims card. |
| 17 how filter pushdown works / verify | Weak | 1 | "reads only the row groups that can hold it" stated; row group never defined; no way to check. |
| 18 sort key | **Good** | 1 | Cards say string order, `Q10` before `Q2`. |
| 19 numeric id | **None** | - | Only `numeric-id` in the release `datavalue`, for entity values. No idiom for the subject `id`. |
| 20 file names / stability | Partial | 2 | Card "Files" gives `part-{i}-of-{n}`; `push.md` shows the `chunks-` names pre-compaction. Nothing says names change per release. |
| 21 pandas gotchas, 22 DuckDB/Spark | **None** | - | Polars/datasets only. |

### C. Languages

| # | Verdict | Hops | Where / what's wrong |
|---|---|---|---|
| 23 fallbacks | **Good** | 1 | Labels card: chain and code. But **only on the labels card**; README defers. |
| 24 what `mul` is | Weak | 2 | README one clause in the example; cards use `mul` in the chain without defining it. Aliases card doesn't mention it. |
| 25 one label per item at scale | Partial | 1 | Labels card code does it over a lazy frame. Memory/perf of the `unique` on `all` items not addressed. |
| 26 `zh*` variants | Partial | 1 | Card mentions variants and script conversion. |
| 27 missing language | **None** | - | Nothing on why a folder wouldn't exist. |
| 28 descriptions `mul` | OK | 1 | Descriptions card chain. |
| 29 monolingualtext | OK | 1-2 | Claims card + README. |
| 30 null aliases dropped | Weak | 3 | README "Notes on coverage" only; neither aliases card says it. Does an item with all-null aliases disappear? Not said. |

### D. Statements

| # | Verdict | Hops | Where / what's wrong |
|---|---|---|---|
| 31 `datavalue` fields per type | **Good** | 1 | Claims card table. |
| 32 time: `+`, precision, BCE, parsing | **None** | - | Fields listed, no semantics, no parse example. |
| 33 quantity: `unit` form, bounds | Weak | 1 | "the unit", but not whether it's a Q id or URL (it must be an id since it joins to `unit-labels`, but that's inferred). |
| 34 coordinates | Partial | 1 | Fields + `precision__integer/number` quirk. |
| 35 explode qualifiers/references | **None** (example) | - | Schema given. No worked example; README says "named the same way". |
| 36 `rank` / "best value" | Weak | 1 | Values listed, no rule for choosing; demos/divisions "using ranks" is the only example (README one line). |
| 37 unknown/no value | **Good** (release) | 1 | Release card: `snaktype`. Main card says not distinguished. |
| 38 dropped statements | Good | 1 | Claims card section, README notes. Says they're not published. |
| 39 statement id / ordering | Partial | 1 | `statement_id` release only. Order: "keep their order". |
| 40 which field for which `datatype` | OK | 1 | Table in card. Double underscore never explained. |
| 41 all `datatype` values | Weak | 1 | Examples only. |
| 42 join item values to names | OK | 2 | Table in claims card; code example only for property. |
| 43 instances incl. subclasses | Weak | 2-3 | Only through `demos/concepts.py` (README paragraph). No plain idiom. |
| 44 badges | Partial | 1 | Release links card. Not in main. |

### E. Local copy, disk

| # | Verdict | Hops | Where / what's wrong |
|---|---|---|---|
| 45 download only some languages via `just download` | **None** | - | README only says it downloads six repos. `snapshot_download` with patterns is the workaround, never framed as "`just download` can't". |
| 46 hub dir / flags | **None** | - | `config.md` lists `HUB_COPY_DIR` under `WORK_DIR`; no flag, not stated as fixed. |
| 47 resume a download | Weak | 3+ | Only in a code docstring + README "resumable"? Docs site: none. |
| 48 how much disk do I need | **Scattered** | 3-4 | README "Running", `setup.md` disk table, `pull.md` prefetch, `tuning.md`. Different numbers apply to different modes. |
| 49 prefetch knobs | Weak | 2 | `config.md` lists `PREFETCH_*` constants; not stated they are not env vars. |
| 50 HF token, other namespace | Partial | 1 | `setup.md` covers login, `HF_TOKEN`, `HF_USER`; "change namespace" = edit source, unstated. |

### F. Running it

| # | Verdict | Hops | Where / what's wrong |
|---|---|---|---|
| 51 limit languages/tables/chunks | **None** | - | No answer, so a user can't tell "unsupported" from "undocumented". |
| 52 change `HF_USER` | Weak | 2 | `setup.md`/`config.md` say it's a constant; no "edit config.py", and functions have an `hf_user` arg the CLIs can't pass. |
| 53 private repos | **Good** | 1 | `setup.md`. |
| 54 state numbers / where | **Good** | 1 | `monitoring.md` table. |
| 55 retry or reset one chunk | **None** | - | Docs say rerun; nothing on resetting one chunk. `reset_run.py` (listed in `scripts.md`) nukes Hub too. |
| 56 `CLEAN_UP_LOCAL` | Good | 2 | `tuning.md`, `config.md`. Code-only toggle not said. |
| 57 RAM | Partial | 1 | `setup.md` "Memory": trial totals, "several GB" per chunk, tuned on 125 GB. No minimum, no OOM symptom. |
| 58 two runs at once | **None** | - | |
| 59 how long | OK | 2 | `performance.md` (measured), linked from guide index. |
| 60 is finalise required / usable without | **Contradictory** | 3 | README vs `source-copy.md`; no "what you have without it". |
| 61 finalise refused | Partial | 2 | `finalise.md` says it refuses; error says "run process-wikidata" and doc doesn't tie them together. |
| 62 skip card push | **None** | - | `cards.md` documents `just cards` (render only) but not a `finalise` without push. |
| 63 card-stats without `hub/` | **None** | - | Only README line that it reads `hub/`. |
| 64 sha mismatch | Weak | 2 | `push.md` says must match; outcome only generic ("halts on any error", `monitoring.md`). |

### G. Releases

| # | Verdict | Hops | Where / what's wrong |
|---|---|---|---|
| 65 `run` vs `release` | Partial | 2 | `source-copy.md` explains it, but is last in the nav and README mixes both. |
| 66 valid release / `latest-dump` | Good | 1 | `release.md`. |
| 67 mirror, smaller dump | Partial | 1 | `WIKIDATA_DUMPS_URL` in `release.md`; other dump variants not discussed. |
| 68 no `lbzip2` | Weak | 1 | Requirement listed in `setup.md`; no alternative or error. |
| 69 deleting the bz2 early/late | Partial | 1 | Instruction ok; consequence of deleting early not said. |
| 70 `previous` argument | Good / gap | 1 | `release.md` explains; first-ever release (empty repos) vs the Justfile's two args: `hub.md` implies yes but doesn't say. |
| 71 skip scholarly | **None** | 4+ | Implied impossible by `finalise.md` + `claims-labels.md` (main's claims_labels needs scholar labels); never stated. |
| 72 finalise "waiting" | **Good** | 1 | `release.md`, `finalise.md`. |
| 73 why the order, interrupts | Good | 1 | Both pages. |
| 74 inspect the build branch | Partial | 2 | Branch name documented; `revision="build-..."` example isn't. "`main` unchanged until promotion" is. |
| 75 promote failure / reversibility | Partial | 1 | Rerun-safe; undo via `previous` tag implied. |
| 76 pin a release | Partial | 1 | `load_dataset(revision=...)` only; `hf://...@rev` and `snapshot_download` not shown. |
| 77 schema differences main vs release | Weak | 4 | Pieces in README / `source-copy.md` / cards; no table, no "will my query break". |
| 78 `WIKIDATA_SCHOLAR` vs recipe arg | Good | 1 | `release.md` table. |
| 79 change scholarly classes | Weak | 2 | `dump.md` + `p31_survey.py`; re-routing already-routed chunks: not said. |
| 80 is `route-release` destructive | Weak | 2 | `dump.md` implies it (split chunk deleted); no warning or undo. |

### H, I. Cards, misc

| # | Verdict | Hops | Where / what's wrong |
|---|---|---|---|
| 81 edit a card | Good | 1 | `cards.md`. |
| 82 stale figures | Good | 1 | `cards.md`, `finalise.md`. |
| 83 license | Weak | 2 | CC0 only in the cards' footer (Hub). README and docs site never say it. |
| 84 how current / update policy | **Contradictory** | 2 | `changelog.md` by date; README stale; no "will this update". |
| 85 redirects/deleted/properties | Partial | 1 | Labels card: items *or properties*. Redirects/deleted: none. |
| 86 full list of differences from Wikidata | Weak | 3 | README notes, claims cards, `source-copy.md`. |
| 87 demo prerequisites | Weak | 1-2 | README: "on a local copy" for one; the `.sh` scripts: none. |
| 88 sae/embedding extras | Weak | 2 | `emb`/`sae` groups only in `pyproject.toml`; `sae/README`. |
| 89 install | Good | 1 | `setup.md`: `uv sync`, Python 3.13. |
| 90 citation | **None** | - | |

## Worst offenders (by how likely a user is to hit them x how bad the dead end is)

1. **README vs docs contradictions** (#8, #10, #60, #84): a first-time user picks the wrong repo/mode.
2. **No consumer documentation on the docs site** (#11-#22, #31-#44): everything lives on Hub
   cards, and the cards omit the practical bits (auth/throttling, pandas/duckdb, time parsing, exploding
   qualifiers, `rank` choice, numeric ids).
3. **"Can I do X?" with no stated answer** (#13, #45, #51, #58, #71): absence reads as "undocumented" not "unsupported".
4. **Source-edit-only knobs** (#46, #49, #52, #56): not labelled, so users hunt for env vars.
5. **`mul` / row group / snak** used without definition on the pages that rely on them (#17, #24).
6. **Release-mode gaps**: skipping the scholarly set (#71), first-ever release (#70), destructive `route-release` (#80), main vs release schema diff (#77).

## Cheapest fixes (to try next)

- One "Using the data" page in the docs site (un-exclude or mirror the cards), with a "which table" decision list, an id-lookup recipe, `mul`, and links per subset.
- Update README to the seven-table / two-set / tag-`20260507` world, and one line saying which build is on `main`.
- A "what is not supported" list: no language/table limiting, no skipping scholarly, one run per directory.
- In `config.md`, mark each constant as env or "edit `config.py`".
- A shared glossary linked from every card.

## After the fixes of 2026-10-04

Docs changes only (journal: `docs/journal/2026-10-04-docs-audit-fixes.md`); `src/` and the
card templates are unchanged, as the run of 20260928 is in progress. "Using" is
`docs/using-the-data.md`, in the nav and linked from the README and the home page. Each
claim was checked against `src/`; the audit's inferences held: no option limits languages,
tables or chunks (`main.run`, `Table`), the scholarly set cannot be skipped
(`main.finalise` waits for the other set's labels), and nothing locks a working directory.

| # | Verdict | Hops | Where |
|---|---|---|---|
| 5 sites | OK | 1 | Using, "Which table" (`enwiki`, `frwikivoyage`, `commonswiki`) |
| 6 Q vs P | OK | 1 | Using, "Which table": `wikidata-entities` (releases) |
| 7 lexemes | OK | 1 | Using, "Which table": not included |
| 8 entities table | Fixed | 1 | README intro, home page: six on `main` now, seven in a release |
| 9 scholarly split | OK | 1 | Using, "Which table" |
| 10 old vs new repos | Fixed | 1 | README intro, home page, Using "Which build `main` holds" |
| 11 subset names | OK | 1 | Using, "Subsets and `all`" (`HfFileSystem().ls`) |
| 12 `all` | OK | 1 | Using: one row per language per id; the only subset of claims and entities |
| 13 several languages | OK | 1 | Using: `data_files` |
| 14 auth, throttling | OK | 1 | Using, "Reading from the Hub" |
| 17 row group | OK | 1 | Using, "Terms" |
| 20 file names change | OK | 1 | Using, "Pinning a release": read `*.parquet` |
| 24 `mul` | Partial | 1 | Using, "Languages and `mul`" and "Terms"; cards still undefined (template change deferred) |
| 32 dates | OK | 1 | Using, "Dates": precision, zeros, BCE, calendar, year example |
| 33 quantity unit | OK | 1 | Using, "Quantities": `1` or the unit's URI |
| 35 qualifiers, references | OK | 1 | Using, examples for both, for each build |
| 36 rank | OK | 1 | Using, "Choosing by rank" |
| 45 `just download` per language | OK | 2 | guide/setup "Limits" |
| 46 hub dir | OK | 2 | reference/config: edit `config.py` |
| 47 resume a download | OK | 2 | guide/setup "Limits" |
| 48 disk | OK | 2 | guide/setup "Disk", both builds |
| 49 prefetch knobs | OK | 2 | reference/config: edit `config.py` |
| 50, 52 other namespace | OK | 1 | guide/setup: edit `HF_USER` in `config.py` |
| 51 limit languages/tables/chunks | OK | 1 | guide/setup "Limits": not supported |
| 55 reset one chunk | OK | 2 | guide/monitoring: no command; redone below `PARTITION`, not past `PROCESS` |
| 56 `CLEAN_UP_LOCAL` | OK | 2 | reference/config, guide/tuning: `config.py` |
| 58 two runs | OK | 1 | guide/setup "Limits": no lock, one run per directory |
| 60 finalise required | Fixed | 1 | README "Running", guide/source-copy; reference/finalise "Before finalise" |
| 61 finalise refused | OK | 1 | reference/finalise: what to run |
| 62 skip card push | OK | 2 | reference/cards: no option |
| 64 sha mismatch | OK | 2 | reference/push, step 3 |
| 68 no `lbzip2` | OK | 1 | guide/setup: no fallback, the error |
| 69 deleting the bz2 early | OK | 1 | guide/release, step 2 |
| 70 first release | OK | 2 | reference/hub |
| 71 skip scholarly | OK | 1 | guide/setup "Limits": not possible |
| 74 inspect the build branch | OK | 2 | reference/hub: `revision=`, `@build-…` |
| 75 promote failure | OK | 2 | reference/hub "If promotion stops" |
| 76 pin a release | OK | 1 | Using: `load_dataset`, `hf://…@rev`, `snapshot_download` |
| 77 schema differences | OK | 1 | Using, "Builds compared" |
| 79, 80 re-route, destructive | OK | 2 | reference/dump "Route" |
| 83 license | OK | 1 | README, Using |
| 84 how current | Partial | 1 | README intro, changelog; no stated update policy |
| 90 citation | OK | 1 | Using: no formal citation |
| 91 find by name | OK | 1 | Using, "Looking things up": reads the whole subset |
| 92 find by external id | OK | 1 | Using, "Looking things up" |
| 93 stable ids | Partial | 1 | release claims card (`statement_id` stable); not on the docs site |
| 94 changes between releases | **None** | - | `entities` has `lastrevid`, `modified`; no page says how to compare |
| 95 truthy version | OK | 1 | Using, "Choosing by rank": no such table, the rule to apply |
| 96 which build a card describes | Partial | 1 | README intro says which build is on `main`; cards follow their branch |

### Still open

- Card templates (deferred until the run of 20260928 is finished; recorded in the journal
  entry): define `mul` in the cards, link the terms list, say on the aliases cards that
  null aliases were dropped (#24, #30).
- Not answered anywhere, and not written here: pandas, DuckDB and Spark (#21, #22), numeric
  ids of the subject (#19), every `datatype` value (#41), transitive subclasses (#43), why a
  language has no folder (#27), redirects and deleted entities (#85), demo prerequisites
  and the `emb`/`sae` groups (#87, #88), comparing releases (#94), RAM minimum and an OOM's
  symptoms (#57).
- Not answerable from the code without running it: what `card-stats` does without `hub/`
  (#63), and what a rerun of `promote-release` does after a repo's commits all landed but
  before its release tag (the copies are committed again; the Hub's answer to a commit
  that changes nothing was not checked).
