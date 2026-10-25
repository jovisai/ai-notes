---
description: Prepare a draft pull request from the current worktree
---
Inspect the current directory, branch, git status, and diff. Summarize changes.
Run the repository's relevant checks and report their results.
Ask before committing or pushing anything the user has not already authorized.
Push the reviewed branch, then create a draft PR with gh pr create --draft.
Describe the behavior change, validation, and any remaining limitations.
Return the PR URL. Leave review requests and merging to the user.
