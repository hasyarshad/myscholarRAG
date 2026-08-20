#!/usr/bin/env bash
# Push this repository to GitHub, using the token in .env.
#
# The token is passed to a single `git push` and never written anywhere. That is
# deliberate: the usual shortcut, `git remote add origin https://TOKEN@github...`,
# stores the credential in .git/config in plain text, where it survives every
# later push, gets copied by anyone who clones your backup, and shows up in
# `git remote -v` output pasted into a bug report.
#
#   ./scripts/push_repo.sh --dry-run
#   ./scripts/push_repo.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

# Always succeeds. Under `set -e` a grep that matches nothing is a failed
# command, and inside an assignment that aborts the script with no message —
# which is exactly what "no .env yet" looks like on a first run.
value() { grep -E "^$1=" .env 2>/dev/null | cut -d= -f2- | tr -d '"'"'"'' || true; }

TOKEN="${GITHUB_TOKEN:-$(value GITHUB_TOKEN)}"
USER_NAME="${GITHUB_USERNAME:-$(value GITHUB_USERNAME)}"
REPO="${GITHUB_RAG_REPO:-$(value GITHUB_RAG_REPO)}"
REPO="${REPO:-myScholar-rag}"
BRANCH="$(git branch --show-current)"

if [[ -z "$TOKEN" || -z "$USER_NAME" ]]; then
    echo "Set GITHUB_TOKEN and GITHUB_USERNAME in .env — see .env.example."
    echo "Fine-grained token, Contents: read and write on $REPO:"
    echo "  https://github.com/settings/tokens?type=beta"
    exit 1
fi

# Refuse to push a dirty tree. A half-committed push is worse than no push,
# and this is the one operation here that other people can see.
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree is dirty. Commit first:"
    git status --short
    exit 1
fi

URL="https://github.com/$USER_NAME/$REPO.git"
echo "$BRANCH -> $URL"
git log --oneline -5

if [[ "${1:-}" == "--dry-run" ]]; then
    echo
    echo "dry run: nothing pushed"
    exit 0
fi

# Create the repository if it is not there yet. Public, because the point is
# that someone can read it.
if ! curl -sf -o /dev/null -H "Authorization: Bearer $TOKEN" \
        "https://api.github.com/repos/$USER_NAME/$REPO"; then
    echo "creating $USER_NAME/$REPO"
    curl -sf -X POST -H "Authorization: Bearer $TOKEN" \
        -H "Accept: application/vnd.github+json" \
        https://api.github.com/user/repos \
        -d "{\"name\":\"$REPO\",\"private\":false,\"description\":\"Adaptive RAG over a corpus of research papers, measured end to end.\"}" \
        > /dev/null
fi

git remote get-url origin >/dev/null 2>&1 || git remote add origin "$URL"
git remote set-url origin "$URL"

git push --set-upstream "https://x-access-token:$TOKEN@github.com/$USER_NAME/$REPO.git" "$BRANCH"

# Restore the tracking ref to the credential-free URL, since the push above set
# it to one containing the token.
git branch --set-upstream-to="origin/$BRANCH" "$BRANCH" >/dev/null 2>&1 || true
git remote set-url origin "$URL"

echo
echo "pushed: $URL"
git remote -v
