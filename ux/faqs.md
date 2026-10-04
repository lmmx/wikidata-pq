# Simulated user FAQs (from src + README only)

Written as a confused user, not as docs. Each one is a spot where I'd go hunting for the
docs because the README/code didn't make it obvious. Source for each is what I could (not)
find in the README, CLI names, env vars and Justfile.

## A. "Which dataset/table do I even want?"

1. I want "the English name of Q42". Is that labels, aliases or descriptions? What's the
   difference between a label and an alias?
2. I want everything Wikidata says about an item (facts like birth date, occupation). Do I
   use claims? Why does claims have `P31` and numbers but no readable text?
3. Where are the human-readable names of properties like `P31`? claims has only ids. Someone
   said claims_labels, but what are `field` and `ref` and why is it per language?
4. What does `field` = `labels` / `property-labels` / `unit-labels` mean in claims_labels? Which
   one do I join on for the item value vs the property vs the unit?
5. I want the Wikipedia page for an item. Is that links? What's `enwiki` vs `en`? How do I get
   Wikipedia in other languages, or Wikivoyage/Commons?
6. Why is there no "type" (item/property/lexeme) column anywhere? How do I tell Q ids from P ids
   in labels?
7. Are lexemes / senses / forms in here? I can't tell from the README.
8. Where are the entity "last modified" / revision / page id fields? README says a seventh
   table `wikidata-entities` exists for releases but the table list shows six. Which do I get?
9. There's `wikidata-labels` and `wikidata-scholar-labels`. Which has Q-items for papers? Why
   can't I find a scholarly article in the normal dataset? Do I need to read both and union?
10. Which repos are philippesaade-derived and which are from the official dump? Are the
    permutans/wikidata-* on `main` the old May 2026 copy or a newer one?

## B. Loading data

11. I ran `load_dataset("permutans/wikidata-labels")` with no config and it either downloaded a
    huge amount or errored. Which subset names are valid? Where's the list?
12. What is `all`? For claims it's the only folder. For labels is `all` every language stacked
    (so duplicates of Q ids across languages)? Should I use it?
13. I want multiple languages at once with `load_dataset`. Can I pass a list? Do I loop and
    concat?
14. I did `pl.scan_parquet(".../en/*.parquet")` on `hf://` and it's slow / asks for a token /
    rate limits. Do I need to log in? Should I download first? How big is `en` for each table?
15. How do I know the row counts or sizes of a subset before pulling it? ("the card gives it" —
    where is the card, I'm on GitHub not the Hub).
16. Can I stream? `load_dataset(streaming=True)` with a language?
17. Does an id filter really only read some of the file? How do I check it's not scanning
    everything? What's a row group and why do I care?
18. Filtering `id == "Q42"` in claims still took a while. Is it sorted by item id or by
    something else? What's the sort key exactly (string sort? so Q10 < Q2)?
19. Why are ids strings like `Q42` and not ints? How do I sort numerically / get the number
    out?
20. What are the file names (`part-{i}-of-{n}.parquet`) and do I ever need to care about them? Do
    they change between releases (so a hard-coded path breaks)?
21. I'm on pandas, not polars. Any gotchas with the struct `datavalue` column?
22. DuckDB/Spark: can I query with the `hf://` path? Does the `datavalue` struct come through?

## C. Languages and fallbacks

23. I want French labels but a lot of items have none. Do I get nothing or the English one?
    How do fallbacks work and where's the chain written?
24. What is `mul`? Why does Q42's name not show up under `en`? Why are aliases only in `mul`?
25. I want "the label in my language, else English, else anything" as one column. README has a
    `first()` helper but only for a fixed list; how do I do it for all items at scale (not one
    id)? Does the unique/sort approach blow memory?
26. Does language code `zh` include `zh-hans`, `zh-hant`, `zh-cn`...? Which subsets exist?
27. What if my language isn't a folder? Is it missing from Wikidata or dropped by the pipeline?
28. Descriptions: why does the README example only use `en`, not `mul`? Do descriptions have
    `mul`?
29. Monolingual text values in claims: why do I have to filter by language manually, and why
    does the example filter `datatype != monolingualtext`?
30. About 10% of aliases are null and dropped. Do I lose items entirely from aliases? So "item
    has no alias row" = no aliases, or = dropped?

## D. Reading statements (claims)

31. I filtered `property == "P31"` and the value is a struct. What's in `datavalue` for each
    datatype? How do I get the Q id vs a string vs a number vs a date?
32. For dates: why `+1952-03-11T00:00:00Z` with a plus sign? What about precision, BCE dates,
    calendar model, year-only dates? Are they parseable by polars directly?
33. For quantities: `amount` is a string? What about `unit` (a URL? a Q id?), upper/lower bound?
34. Coordinates: where are lat/lon? Are they in `datavalue` fields? What's globe?
35. Where are qualifiers (e.g. "start time" on a position held) and references? README says "named
    the same way" but where are they in the row, and how do I explode them?
36. What does `rank` mean? Do I filter to `preferred`? What if there is no preferred? Are
    `deprecated` statements included? How do I get "best" value per property like Wikidata UI?
37. "Unknown value" / "no value" snaks: README says the philippesaade copy dropped snaktype.
    So I can't tell "unknown" from "none" in `main`? What happens to those rows — null value?
38. A statement vanished. Notes say snaks on deleted properties and statements whose main value
    is one are dropped. How would I know which ones? Where is the quarantine, and is it
    published?
39. What's the statement id? Can I tell two statements with same property+value apart? Is the
    order of statements/qualifiers preserved?
40. `datavalue__string` in the example, vs `text`, `id`, `amount`, `time`. Is there a table of
    which field goes with which `datatype`? Why the double underscore name?
41. What are all possible `datatype` values (`wikibase-item`, `external-id`, `commonsMedia`,
    `url`, `globe-coordinate`, ...)? How do I find every one that exists?
42. How do I follow an item-valued claim to its label at scale (join on `datavalue.id`)?
    Which table has the labels for referenced items, labels or claims_labels, and why can't I
    just use labels?
43. How do I get "all instances of X" (P31 = X) including subclasses (P279 transitive)? Is
    there a helper? `demos/concepts.py`... but what's the idiom in plain polars?
44. How do I resolve sitelinks / badges (e.g. featured article)? Dropped in main?

## E. Local copy and disk

45. "Downloading only what you need" shows a `snapshot_download` with allow_patterns. What if I
    want 3 languages of claims_labels, or one language of everything? Is `just download` only
    all-six-full-repos (35.5 GB)? Can I pass languages to it?
46. `just download` wrote to `hub/`. Can I change the directory? Is there a flag? Env var?
47. It was interrupted. Is it safe to rerun? Does it re-download everything or verify sha256?
48. I have 100 GB free, is that enough for `just run`? README gives 60 GB prefetch + 25 GB +
    pause below 100 GB free. Which number do I plan for? What happens when it pauses — does it
    hang or fail?
49. Can I run with less disk by lowering prefetch? Where are the knobs
    (`PREFETCH_BUDGET_GB` etc.)? Only code constants, not env vars?
50. Do I need a Hugging Face token to just read? To run the pipeline? Where do I put it
    (`HF_TOKEN`, `huggingface-cli login`)? Does it need write on `permutans`? (I'm not
    permutans — how do I push to my own namespace?)

## F. Running the pipeline myself

51. I want to rebuild this for myself with fewer languages. Is there an option to limit
    languages / tables / chunks? (Looks like no. Do I just filter after?)
52. `HF_USER = "permutans"` is hardcoded in `config.py`. How do I change the target namespace
    without editing source? Is it an env var? (`hf_user` arg exists on functions but the CLI
    entry points take no args.)
53. `HF_REPO_PRIVATE = False` — I don't want to publish publicly. How do I switch? Code-only?
54. What does `just run` do if I Ctrl-C? Resume semantic: "state 0..6" — what do the numbers
    mean and how do I inspect where chunk 123 is? Where's `state/`?
55. A chunk failed. How do I retry just that one? How do I reset state for one chunk? Do I
    delete the JSONL line?
56. What's `CLEAN_UP_LOCAL` and can I set it from env to keep intermediate files for debugging?
    It's a code constant too.
57. `WIKIDATA_WORKERS` default 6 — memory per worker? Claims can exceed 1M chars per field. How
    much RAM do I need? What does an OOM look like and is it resumable?
58. Is the process safe to run twice on the same directory (two terminals)? Locks?
59. How long does a full run take? Any progress display / ETA beyond log lines?
60. `just run` vs `just finalise` — do I have to run finalise? What does skipping it leave me
    with (many ~group files, unsorted)? Is the result still usable?
61. `finalise` refused: "Chunks are not all complete: run process-wikidata" or "A group is not
    yet uploaded". What does that mean in practice, what do I run?
62. `finalise` pushes cards — I don't want my repos' README changed. Can I skip that? `just
    cards` renders locally, but does `just finalise` have a no-push mode?
63. `just card-stats` needs `hub/`. If I don't have the local copy, what's the error and what do
    I do first?
64. Verification: "verify it by sha256". What happens on a mismatch — retried, aborted,
    silently continues? Where is it logged?

## G. Releases from official dumps

65. What's the difference between `just run` (philippesaade copy) and `just release`? Which
    one do I want for "latest Wikidata"?
66. `WIKIDATA_RELEASE=20260928` — what date format, and how do I find valid dates?
    (`just latest-dump`: does it print or set something? What's "full JSON dump"?)
67. `just download-dump` is 103 GB. How much disk in total? Can I use the smaller `latest-all`
    variants / truthy dumps / a mirror? (`WIKIDATA_DUMPS_URL` exists — where's it said?)
68. `split-dump` needs `lbzip2`. What if I don't have it — error message? Alternative (`pbzip2`,
    `bzip2`)? How slow?
69. After split, README says delete the bz2 by hand "once split.done is written". What if I
    delete too early? What if I forget and run out of disk (210 GB)?
70. `just release 20260928 20260507` — what is the second arg? Previous release for tagging?
    What if this is my first release (no previous)? Is it required?
71. What's the scholarly / main split for? Can I skip scholarly (it's huge, I don't want
    papers)? If I skip it, does `claims_labels` for main still work ("needs the scholarly set's
    labels", finalise stops before claims_labels)?
72. `finalise-release` "stops before claims_labels... run it again". Is that an error or
    expected? I ran it and it just returned — how do I know it's waiting vs done?
73. Order of steps 1-5 is `scholar` first then `main`, then finalise main, scholar, main
    again. Why? Can I just run `just release` and trust it? What do I do if I interrupt in the
    middle of step 3?
74. `build-20260928` branch: how do I look at the data before it's promoted? Via
    `revision="build-20260928"`? Will `main` readers see it half-built?
75. What does `promote-release` do to my users if it fails halfway? Is it atomic? Is it
    reversible (previous tag)?
76. After promotion, how do users pin a release: `revision="20260928"` — for `load_dataset`, for
    `hf://` paths (`@20260928`), for `snapshot_download`?
77. Is a release's schema the same as `main`'s? README says release has extra fields and a
    `null` datatype for deleted properties. Will my existing queries break moving between
    them? Is there a diff of schemas?
78. Why does `just release` env var `WIKIDATA_SCHOLAR` exist if the recipes take `main` or
    `scholar`? Which wins? Do I set both?
79. How do I add classes to the scholarly set (`SCHOLARLY_CLASSES`)? Edit source and rerun
    `route-release`? Does it re-route already-routed chunks?
80. `just route-release` — does it move the scholarly rows out of main chunks (destructive)?
    Can I undo?

## H. Cards and docs

81. What's in "dataset cards" and where do I edit one? `docs/dataset_cards` templates ->
    rendered where? What variables can I use?
82. Cards figures stale after a new release — what regenerates them, `card-stats` or
    `finalise` or both? Where do the numbers (files/bytes/rows per language) come from?

## I. Misc / expectations

83. License? Wikidata is CC0 — does that hold for the derived Parquet? Is philippesaade's
    dataset license carried over?
84. How current is this? "2026-05-07 without scholarly articles" — will it be updated? Is
    there a changelog per release I can read?
85. Is `Q42` always in the dataset? Are redirects and deleted items included? What about
    properties (`P31`) — are they rows in labels too?
86. Any known differences from Wikidata itself (e.g. dropped statements, truncated
    sitelinks)? Where's the complete list beyond "Notes on coverage"?
87. Demos: do they need the local copy or run over the Hub? `demos/item.py --data wikidata`
    implies local. What do the shell scripts (`bbc_things.sh` etc.) assume exists
    beforehand?
88. SAE/embedding-atlas bits (`sae/`, `embedding-atlas` extra, `emb`/`sae` dependency groups):
    which extras do I install (`uv sync --group emb`?) and are they needed for the main
    tools?
89. Install: how do I get the `process-wikidata`, `download-wikidata` commands? `pip install`
    from where, `uv sync`? Python version? Is it on PyPI?
90. How do I cite this?
