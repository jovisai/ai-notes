#!/usr/bin/env bash
# Run from the main checkout (or one of its subdirectories).
# Usage: bash cw.sh setup <branch> [sparse-path]
#        bash cw.sh list
#        bash cw.sh cleanup <branch>
# Optional: CW_COPY_ENV=1 copies the main checkout's .env on setup.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
command=${1:-help}
branch=${2:-}

if [[ "$command" == list ]]; then
    git -C "$repo_root" worktree list
    exit 0
fi

if [[ "$command" != setup && "$command" != cleanup ]]; then
    printf 'Usage: bash cw.sh {setup <branch> [sparse-path]|list|cleanup <branch>}\n' >&2
    exit 1
fi
if [[ -z "$branch" ]] || ! git check-ref-format --branch "$branch" >/dev/null; then
    printf 'Supply a valid branch name.\n' >&2
    exit 1
fi

# Absolute paths stay correct after entering a nested branch such as feature/auth.
worktree_base="$(dirname "$repo_root")/$(basename "$repo_root")-worktrees"
worktree_path="$worktree_base/$branch"

case "$command" in
    setup)
        mkdir -p "$(dirname "$worktree_path")"
        options=()
        if [[ -n "${3:-}" ]]; then options+=(--no-checkout); fi
        if git -C "$repo_root" show-ref --verify --quiet "refs/heads/$branch"; then
            git -C "$repo_root" worktree add "${options[@]}" "$worktree_path" "$branch"
        else
            git -C "$repo_root" worktree add "${options[@]}" -b "$branch" "$worktree_path" HEAD
        fi
        if [[ -n "${3:-}" ]]; then
            git -C "$worktree_path" sparse-checkout init --cone
            git -C "$worktree_path" sparse-checkout set -- "${3}"
            git -C "$worktree_path" checkout "$branch"
        fi
        if [[ "${CW_COPY_ENV:-0}" == 1 && -f "$repo_root/.env" ]]; then
            cp "$repo_root/.env" "$worktree_path/.env"
        fi
        # Tracked CLAUDE.md and .claude files are already checked out by Git.
        # Never recursively copy .claude: it may contain other worktrees.
        printf 'Created %s\nInstall project dependencies there, then run claude.\n' "$worktree_path"
        ;;
    cleanup)
        # Git refuses dirty worktrees; the branch is only considered after removal.
        git -C "$repo_root" worktree remove "$worktree_path"
        if ! git -C "$repo_root" branch -d "$branch"; then
            printf 'Worktree removed; branch retained. Inspect it before deleting.\n' >&2
        fi
        ;;
esac
