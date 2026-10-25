# About these docs

These pages describe what the code does and why, as of the commit each page names. They are written by hand, and the code may since have changed. To make that visible, each reference page ends with a **Documented against** block. It lists the commit the page was written against and the SHA-256 of each file it describes, as the file was at that commit.

## Checking whether a page is out of date

Compare the recorded checksums with the files as they are now:

```sh
sha256sum src/wikidata/dump.py src/wikidata/scholarly.py
```

A file whose checksum differs has changed since the page was written. To see how:

```sh
git diff 6243a2c -- src/wikidata/dump.py
```

where `6243a2c` is the commit in the page's block. Update the page, then record the new commit and checksums in its block.

## Where things are written down

| Place                                                                                | Holds                                                                                     |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| These pages (`docs/`, outside `journal/`)                                            | what the code does now, and the intent behind it                                          |
| `docs/journal/`                                                                      | dated entries of investigations, measurements and decisions (format in `docs/JOURNAL.md`) |
| [Changelog](https://lmmx.github.io/wikidata-pq/pr-preview/pr-165/changelog/index.md) | what changed, by date                                                                     |
| `README.md`                                                                          | the published datasets, for their users                                                   |
| `DESIGN.md`                                                                          | the original design of the philippesaade build's steps                                    |
| Docstrings                                                                           | each function's contract                                                                  |

## Building the site

```sh
just mkdocs          # serve locally
just mkdocs build    # build into site/
```

The recipe runs mkdocs with `uvx` and the packages in `docs/docs-packages.txt`.

The site deploys to GitHub Pages, at <https://lmmx.github.io/wikidata-pq/>: `.github/workflows/site.yml` runs the same `mkdocs build` into `site/` and pushes it to the `gh-pages` branch on every push to `master`. A pull request that changes the docs gets a preview under `pr-preview/pr-<number>/`, linked from a comment on the pull request and removed when it closes. The packages are installed on their own, without the project, which needs Python 3.13 and the pipeline's dependencies.
