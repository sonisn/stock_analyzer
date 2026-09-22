#!/usr/bin/env bash
# Cron: move this checkout to origin/main only if the tests pass there.
#
# The old `git pull` put whatever reached main into the next money-advice
# email an hour later, tested or not. This runs the suite on the new commit
# in a throwaway worktree first and fast-forwards only on a pass; a failure
# exits non-zero, which run_job.sh turns into an alert email.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

git fetch --quiet origin main
if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ]; then
    echo "already at origin/main ($(git rev-parse --short HEAD))"
    exit 0
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "working tree has local changes — not updating"
    exit 1
fi

target="$(git rev-parse --short origin/main)"
tree="$(mktemp -d)"
trap 'git worktree remove --force "$tree" >/dev/null 2>&1 || rm -rf "$tree"' EXIT
git worktree add --quiet --detach "$tree" origin/main

echo "testing $target before updating"
if ! (cd "$tree" && uv run --quiet --extra dev pytest -q -x -p no:cacheprovider); then
    echo "tests failed on $target — staying on $(git rev-parse --short HEAD)"
    exit 1
fi
git merge --ff-only --quiet origin/main
echo "updated to $target"
