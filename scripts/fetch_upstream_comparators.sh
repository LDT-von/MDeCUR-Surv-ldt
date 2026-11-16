#!/usr/bin/env bash
# Fetch the four upstream comparator repositories at their pinned commits.
#
# The upstream trees are NOT part of this repository (they are third-party
# artifacts).  The server must clone them once, at the exact commits recorded
# in comparison_sources.py, before any upstream arm can run.
#
# The paths are derived from this script's own location, so the layout matches
# what comparison_sources.py computes:
#
#   <code dir>/scripts/fetch_upstream_comparators.sh
#   <code dir>/comparison_sources.py   ->  COMPARATORS = <code dir>/../papers/comparators
#
# The script is deliberately conservative:
#   * it never modifies an existing checkout;
#   * it refuses to continue when a checkout has a dirty working tree;
#   * it verifies the recorded per-file SHA256 after the fetch;
#   * it never writes into the upstream trees.
set -euo pipefail

CODE_DIR="${COMPARATOR_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
COMPARATORS="${COMPARATOR_ROOT:-$(cd -- "$CODE_DIR/.." && pwd)/papers/comparators}"

PIBD_URL=https://github.com/zylbuaa/PIBD
PIBD_COMMIT=bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78
CAMR_URL=https://github.com/wxq-ustc/CAMR
CAMR_COMMIT=8fbae0a0bc4e9da31547b5eab5d773e6b2049e93
DEREF_URL=https://github.com/ZJUMAI/DeReF
DEREF_COMMIT=6bdbf81e76d1c7fa1d49b05f1a8fec2fbfc7b6d3
DIMAF_URL=https://github.com/Trustworthy-AI-UU-NKI/DIMAF
DIMAF_COMMIT=3fe81236aafac88977b730745314b8c9906cb4d3

printf 'Code directory: %s\n' "$CODE_DIR"
printf 'Upstream root:  %s\n' "$COMPARATORS"

mkdir -p -- "$COMPARATORS"

fetch_one() {
  local name=$1 url=$2 commit=$3
  local target="$COMPARATORS/$name"
  if [[ -e "$target" ]]; then
    printf '%s already exists; verifying the pinned commit\n' "$name"
    if [[ ! -d "$target/.git" ]]; then
      printf '%s is not a git checkout; refusing to overwrite it\n' "$name" >&2
      return 2
    fi
    if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
      printf '%s has uncommitted changes; refusing to touch it\n' "$name" >&2
      return 2
    fi
    local current
    current=$(git -C "$target" rev-parse HEAD)
    if [[ "$current" != "$commit" ]]; then
      printf '%s is at %s, not the pinned %s\n' "$name" "$current" "$commit" >&2
      printf 'Fetch the pinned commit yourself and report; do not reset a dirty tree.\n' >&2
      return 2
    fi
  else
    printf 'Cloning %s\n' "$name"
    git clone --quiet --no-tags "$url" "$target"
    git -C "$target" fetch --quiet --no-tags origin "$commit"
    git -C "$target" checkout --quiet "$commit"
  fi
  local head
  head=$(git -C "$target" rev-parse HEAD)
  printf '%s at %s\n' "$name" "$head"
  if [[ "$head" != "$commit" ]]; then
    printf '%s does not match the pinned commit %s\n' "$name" "$commit" >&2
    return 2
  fi
}

fetch_one PIBD "$PIBD_URL" "$PIBD_COMMIT"
fetch_one CAMR "$CAMR_URL" "$CAMR_COMMIT"
fetch_one DeReF "$DEREF_URL" "$DEREF_COMMIT"
fetch_one DIMAF "$DIMAF_URL" "$DIMAF_COMMIT"

printf 'Verifying the recorded per-file hashes\n'
cd -- "$CODE_DIR"
python - <<'PY'
import comparison_sources as sources
hashes = sources.verify_upstream()
for name, files in sorted(hashes.items()):
    print(name, len(files), 'files verified')
print('all upstream checkouts match the recorded hashes')
PY
