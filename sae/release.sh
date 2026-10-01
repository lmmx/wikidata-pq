# Sourced by the sae/*.sh runners (after `dir`, a run's folder, is set, where there is one):
# which release's local copy (`data`) and identifier sets (`inputs`) a run uses. RELEASE=
# names a release (a Wikidata dump date, see src/wikidata/config.py); a run trained with one
# records it in $dir/release, which its later steps read. With neither, the build from
# philippesaade/wikidata: hub/ and sae/output/.
release=${RELEASE:-}
if [ -n "${dir:-}" ] && [ -f "$dir/release" ]; then
  recorded=$(<"$dir/release")
  if [ -n "$release" ] && [ "$release" != "$recorded" ]; then
    echo "$dir was trained on release $recorded, not $release" >&2
    exit 1
  fi
  release=$recorded
fi
if [ -n "$release" ]; then
  inputs=sae/output/releases/$release
  data=${DATA:-releases/$release/hub}
else
  inputs=sae/output
  data=${DATA:-hub}
fi
