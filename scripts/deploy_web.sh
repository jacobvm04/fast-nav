#!/usr/bin/env bash
# Publish web/ to the public demo repo (github.com/jacobvm04/fast-nav-demo -> GitHub Pages).
#
#   scripts/deploy_web.sh             # sync web/ as-is, commit, push
#   scripts/deploy_web.sh --export    # re-run scripts/export_web.py first (needs checkpoints)
#
# The demo repo's README.md is owned by the demo repo and never overwritten.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEMO="${DEMO_DIR:-$ROOT/../fast-nav-demo}"

if [[ "${1:-}" == "--export" ]]; then
  (cd "$ROOT" && uv run python scripts/export_web.py)
fi

(cd "$ROOT" && node web/test/parity.mjs)

rsync -a --delete \
  --exclude .git --exclude README.md --exclude .nojekyll --exclude CNAME \
  "$ROOT/web/" "$DEMO/"

cd "$DEMO"
if git status --porcelain | grep -q .; then
  git add -A
  git commit -m "Sync demo from fast-nav web/ ($(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo local))"
  git push
  echo "deployed — live in ~1 min at https://jacobvm04.github.io/fast-nav-demo/"
else
  echo "no changes to deploy"
fi
