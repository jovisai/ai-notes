---
title: "The Complete Guide to Git Worktrees with Claude Code"
date: 2026-02-02
lastmod: 2026-10-25
description: "Run parallel Claude Code sessions with Git worktrees. Start in 60 seconds, set up each environment, review and merge changes, and fix common errors."
articleTools: true
---

## Start in 60 Seconds {#1-introduction}
{{< anchors ids="the-problem-context-switching-kills-productivity the-solution-git-worktrees step-1-create-worktrees-for-each-task step-3-open-multiple-terminal-panes running-claude-in-a-worktree" >}}
Git worktrees give each Claude Code session its own working directory and branch. You can build a feature and fix a bug in parallel without stashing one task to start another.

From an existing Git repository, open two terminals at the repository root:

```bash
# Terminal 1
claude --worktree feature-auth

# Terminal 2
claude --worktree bugfix-api
```

Claude creates these under `.claude/worktrees/`, with branches named `worktree-feature-auth` and `worktree-bugfix-api`. Add `.claude/worktrees/` to `.gitignore`. If workspace trust has not been accepted, run `claude` once in the repository first.

Check availability with `claude --help`. If your installation lacks `--worktree`, use the [manual Git setup](#4-basic-setup-and-commands). The [official worktree documentation](https://code.claude.com/docs/en/worktrees) covers the current behavior.

## On This Page {#table-of-contents}

[How isolation works](#2-understanding-git-worktrees) · [Complete workflow](#5-parallel-development-workflow) · [Manual setup](#4-basic-setup-and-commands) · [Troubleshooting](#10-troubleshooting) · [Advanced recipes](#8-advanced-techniques) · [Command reference](#quick-reference-card)

## How Worktree Isolation Works {#2-understanding-git-worktrees}
{{< anchors ids="what-is-a-git-worktree architecture key-properties 3-why-worktrees--claude-code the-context-preservation-problem benefits-of-worktree--claude-combination the-developers-role-shift" >}}
A linked worktree has its own files, index, and checked-out branch. Its `.git` file points back to the shared repository data. Commits and remotes are shared; edits to working files stay in their directory.

{{< figure src="/images/git-worktrees/isolation.svg" link="/images/git-worktrees/isolation.svg" alt="The main checkout and two task worktrees have separate files and Claude sessions but share Git history and remotes. Ports and external services still need coordination." >}}

A separate session keeps each task's conversation focused. Worktrees do not create a sandbox: sessions can still reach shared services, databases, or files outside their directory according to their permissions. Give each dev server its own port, and plan database changes together.

## A Complete Parallel Development Workflow {#5-parallel-development-workflow}
{{< anchors ids="step-by-step-workflow 11-best-practices dos- donts- workflow-checklist" >}}
Start with two independent tasks. If the second task depends on the first task's code, finish or integrate that prerequisite before starting it. Increase concurrency only while you can review the results.

{{< figure src="/images/git-worktrees/workflow.svg" link="/images/git-worktrees/workflow.svg" alt="Create a worktree, set up its environment, give Claude one task, review and test, merge approved work, and clean up after preserving the changes." >}}

### Set Up Each Environment {#step-2-initialize-each-worktree-environment}

Run the project's setup in each worktree. For a Node project with a lockfile, this might be `npm ci`; Python projects need their own environment and dependency installation. Check the repository's instructions first.

Tracked `CLAUDE.md` and `.claude/` files come with a normal checkout. You do not need to run `/init` every time. For native Claude-created worktrees, `.worktreeinclude` can copy selected gitignored files such as `.env`; keep those files ignored and use appropriate local credentials. [Environment setup reference](https://code.claude.com/docs/en/worktrees#copy-gitignored-files-into-worktrees)

### Give Each Session One Task {#step-4-assign-tasks-to-each-claude-session}

Specify the scope, expected behavior, and validation. For example:

```text
Fix the rate-limit bug in issue #456. Read the issue and repository
instructions first. Keep changes within the API limiter and its tests.
Run the relevant checks, then summarize the diff and remaining concerns.
```

Keep branch and session names recognizable. Visit each terminal to answer questions and review progress. Parallel sessions still share your machine's resources and usage limits.

### Review, Merge, and Clean Up {#9-merging-and-integration}
{{< anchors ids="step-5-cycle-through-and-monitor step-6-review-and-merge" >}}
Review each diff and run its relevant tests before merging. A passing branch can still conflict with another branch's changes, so test the combined result too.

The following example uses the branches from the quick start. Run it inside the feature worktree after reviewing its changes; GitHub CLI requires authentication:

```bash
git status
git diff
# Run this project's tests, then commit the reviewed files.
git push -u origin HEAD
gh pr create --draft --title "feat: add authentication"
```

Use your normal PR review and merge process. Worktrees do not merge branches automatically.

#### Cleanup {#post-merge-cleanup}
{{< anchors ids="step-7-cleanup removing-worktrees" >}}
When leaving an interactive `--worktree` session, read Claude's keep/remove prompt. Keep the worktree if you need to return to it. Non-interactive `-p` runs do not perform that exit cleanup. [Cleanup behavior](https://code.claude.com/docs/en/worktrees#clean-up-worktrees)

For manual cleanup, exit the session and return to the main checkout. After the branch's work is preserved, check its status and remove its directory:

```bash
git -C .claude/worktrees/feature-auth status --short
git worktree remove .claude/worktrees/feature-auth
git branch -d worktree-feature-auth
```

Git normally refuses dirty worktrees. If branch deletion is refused after a squash merge, inspect the branch before deciding whether to delete it. Do not add `--force` just to make cleanup succeed.

{{< details id="integration-branch-strategy" summary="Optional: integration branches, cherry-picking, and conflicts" >}}

Use an integration branch when several completed tasks need testing together. From a clean main checkout, create `integration` from your base branch, merge the task branches one at a time, and run the combined test suite. Review that result before merging it through your team's normal process.

#### Cherry-Picking Specific Commits {#cherry-picking-specific-commits}

Use `git cherry-pick <commit>` on the destination branch when you want one reviewed commit rather than the whole task branch. Commit IDs are visible across worktrees because history is shared.

#### Handling Merge Conflicts {#handling-merge-conflicts}

Resolve conflicts on the integrating branch, review the resolved diff, and rerun affected tests. `git merge --abort` cancels an unfinished merge when you need to reconsider the integration. Claude can help explain a conflict, but the final behavior still needs review.

{{< /details >}}

## Manual Git Setup {#4-basic-setup-and-commands}
{{< anchors ids="creating-worktrees listing-worktrees directory-organization" >}}
Use plain Git when you need a specific base branch, an existing branch, or a directory outside Claude's default location. In this example, run from your main checkout and replace `main` if your repository uses another base:

```bash
git worktree add -b feature/auth ../project-auth main
cd ../project-auth
# Install this project's dependencies here.
claude
```

For an existing branch, use `git worktree add ../project-auth feature/auth`. Each branch normally belongs to one worktree at a time. For read-only inspection or experiments, `git worktree add --detach ../project-review HEAD` creates a detached checkout.

Keep sibling worktrees beside the main project, or ignore the directory if you put them inside it. Manual worktrees have the same review and cleanup needs as the native workflow; substitute their actual paths and branch names in the commands above.

## Troubleshooting {#10-troubleshooting}
{{< anchors ids="common-issues-and-solutions" >}}

### “Branch Is Already Checked Out” {#fatal--is-already-checked-out}

Run `git worktree list` to find the checkout using it. Use that directory, create a new branch, or use a detached worktree for inspection. Avoid forcing the same branch into two active worktrees.

### Dependencies Not Found {#dependencies-not-found-module-not-found}

Install dependencies inside the worktree. Untracked directories such as `node_modules` and `.venv` do not arrive with a normal Git checkout. Use the same package manager and lockfile as the main project.

### Port Already in Use {#port-already-in-use}

Assign a different port to each server. For applications that read `PORT`, use `PORT=3001 npm run dev` in one worktree and `PORT=3002 npm run dev` in another. Other frameworks need their own port flags.

### Missing `.env` Files {#env-files-missing}

Gitignored files are absent from a normal worktree. Use `.worktreeinclude` for native creation, or explicitly copy the needed file from the main checkout. Check whether copied credentials point to a shared database before running migrations.

### Deleted Worktree Directory {#worktree-directory-deleted-without-cleanup}

Inspect `git worktree prune --dry-run` before `git worktree prune` to clear stale registrations. Pruning metadata does not recover files from a deleted directory.

### Submodules {#submodules-in-worktrees}

Run `git submodule update --init --recursive` inside the worktree when the project needs submodules. Check their state as part of review and cleanup too.

### Diagnostic Commands {#diagnostic-commands}

```bash
git worktree list --porcelain
git status --short
git worktree prune --dry-run
```

If a worktree is locked, stop the session or process using it before investigating the lock. Check `git worktree list --porcelain` for the reason; unlock only when it is no longer in use.

## Advanced Recipes {#8-advanced-techniques}

Open the recipe you need. The core workflow above is enough for most feature and bug-fix sessions.

{{< details id="6-automation-scripts-and-tools" summary="Automation scripts and tools" >}}


### A Small Shell Shortcut {#simple-bash-function-for-worktree-management}

For native worktrees, a shell function is usually sufficient:

```bash
cw() { claude --worktree "${1:?Usage: cw <task-name>}"; }
```

### Downloadable Worktree Manager {#advanced-worktree-manager-script}

[Download cw.sh](/downloads/git-worktrees/cw.sh) when you need manual Git worktrees with optional sparse checkout. Save it locally and run it from the main checkout or one of its subdirectories:

```bash
bash /path/to/cw.sh setup feature/auth
bash /path/to/cw.sh list
bash /path/to/cw.sh cleanup feature/auth
```

The script resolves absolute source paths before creating nested directories, stops when creation fails, and refuses dirty cleanup. It places worktrees in a sibling directory named `<project>-worktrees`. Set `CW_COPY_ENV=1` only when you want it to copy the main checkout's `.env`. Tracked Claude configuration comes from Git; the script never recursively copies `.claude/worktrees`.

{{< /details >}}

{{< details id="7-custom-claude-commands-for-worktrees" summary="Custom Claude commands for worktrees" >}}

### Creating a Worktree {#create-a-worktree-command}

Use `claude --worktree <task-name>` instead of maintaining a custom creation command. Add project-specific setup only when the native behavior does not cover it.

### Preparing a PR {#create-a-pr-command-for-worktrees}

[Download the draft-PR command](/downloads/git-worktrees/pr.md) and save it as `.claude/commands/pr.md` for `/pr`. It reviews the current worktree, reports validation, and prepares a draft PR within the user's authorization.

### Preparing a Handoff {#create-a-done-command}

[Download the handoff command](/downloads/git-worktrees/done.md) as `.claude/commands/done.md` for `/done`. It checks the branch and diff, runs relevant checks, and reports what is ready for review. It keeps the worktree available until its work is preserved.

{{< /details >}}

{{< details id="sparse-checkout-for-monorepos" anchors="example-5-monorepo-sparse-checkout" summary="Sparse checkout for a monorepo" >}}

Limit the checkout to the packages you need, including shared dependencies. From the main checkout:

```bash
git worktree add --no-checkout -b fix/api ../project-api main
git -C ../project-api sparse-checkout init --cone
git -C ../project-api sparse-checkout set packages/api packages/shared
git -C ../project-api checkout fix/api
cd ../project-api
# Follow the monorepo's dependency setup, then start Claude.
claude
```

Cone mode can retain root-level files. Sparse checkout reduces checked-out content; it does not guarantee better model accuracy. Ensure the selected directories include the configuration and tests your task needs.

{{< /details >}}

{{< details id="12-real-world-examples" anchors="running-multiple-models-in-parallel batch-processing-with-worktrees" summary="Examples: urgent fixes, competing designs, PR review, and migrations" >}}

### Feature Development and an Urgent Bug Fix {#example-1-feature-development--urgent-bug-fix}

Leave the feature session running. Open another terminal at the main repository and run `claude --worktree hotfix-payment`. Give it the issue and a bounded task, then review and merge through the same process as other changes.

### Parallel Feature Comparison {#example-2-parallel-feature-comparison}

Create `search-postgres` and `search-elasticsearch` worktrees. Give both sessions the same acceptance criteria, then compare behavior, operational cost, and measured performance before choosing an implementation. Different prompts or models can be tested this way; neither result is automatically the winner.

### Code Review in Isolation {#example-3-code-review-in-isolation}

With GitHub CLI authenticated, fetch a PR into a local branch and open a detached review worktree. Substitute the PR number and an unused branch name:

```bash
gh pr checkout 423 --branch review/pr-423
git switch main
git worktree add --detach ../project-pr-423 review/pr-423
cd ../project-pr-423
claude "Review this PR and report findings without editing files"
```

Run the checkout and switch commands from a clean main checkout. Prefer fetching a PR into a review branch directly if your team uses a custom Git hosting workflow.

### Large-Scale Migration {#example-4-large-scale-migration}

List the files, divide them into non-overlapping batches, and give each session its own worktree. Put the migration plan in tracked files so every checkout has it. Start with a small batch, inspect its result, and integrate incrementally. Launching an interactive CLI in a shell loop does not make the sessions run in parallel.

{{< /details >}}

{{< details id="container-isolation-with-worktrees" summary="When you also need container isolation" >}}

Worktrees separate working files; containers can additionally separate processes and dependencies. Use a maintained, project-specific development container with the CLI installed. Check the mounted directory, credentials, network access, and shared services. Worktrees alone do not provide those boundaries.

{{< /details >}}

## Quick Reference {#quick-reference-card}
{{< anchors ids="essential-commands" >}}

| Task | Command |
|---|---|
| Start a named Claude worktree | `claude --worktree feature-auth` |
| Create a manual branch from main | `git worktree add -b feature/auth ../project-auth main` |
| Open an existing branch | `git worktree add ../project-auth feature/auth` |
| Inspect all worktrees | `git worktree list` |
| Remove a clean worktree | `git worktree remove <path>` |
| Delete a merged branch | `git branch -d <branch>` |
| Preview stale registrations | `git worktree prune --dry-run` |

### Resume a Session {#session-management}

Use `/resume` inside Claude or `claude --resume` from the repository. Choose the intended task session; returning to a branch alone does not restore a conversation. For session selection flags, see the [CLI reference](https://code.claude.com/docs/en/cli-reference).

## Video Walkthrough

{{< video-preview id="HrNtlJGvvqg" poster="/images/git-worktrees/video-preview.jpg" title="Git worktrees with Claude Code" >}}

## Resources {#resources}

- [Claude Code worktree documentation](https://code.claude.com/docs/en/worktrees): native creation, environment setup, and cleanup.
- [Git worktree reference](https://git-scm.com/docs/git-worktree): manual worktrees, branch rules, and maintenance.
- [Git sparse-checkout reference](https://git-scm.com/docs/git-sparse-checkout): selecting directories in a large repository.
- [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference): session and command-line options.
