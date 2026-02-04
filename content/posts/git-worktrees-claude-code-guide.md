---
title: "Git Worktrees with Claude Code: The Complete Guide"
date: 2026-02-02
---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Understanding Git Worktrees](#2-understanding-git-worktrees)
3. [Why Worktrees + Claude Code](#3-why-worktrees--claude-code)
4. [Basic Setup and Commands](#4-basic-setup-and-commands)
5. [Parallel Development Workflow](#5-parallel-development-workflow)
6. [Automation Scripts and Tools](#6-automation-scripts-and-tools)
7. [Custom Claude Commands for Worktrees](#7-custom-claude-commands-for-worktrees)
8. [Advanced Techniques](#8-advanced-techniques)
9. [Merging and Integration](#9-merging-and-integration)
10. [Troubleshooting](#10-troubleshooting)
11. [Best Practices](#11-best-practices)
12. [Real-World Examples](#12-real-world-examples)

---

## 1. Introduction

### The Problem: Context Switching Kills Productivity

You're deep in the zone with Claude Code, building a complex feature. Claude follows your plan, understands your codebase perfectly, and you're making incredible progress. Then suddenly—a critical bug report comes in, or a PR needs urgent review.

Traditional workflow:
```bash
git stash push -u -m "WIP: feature work"
git checkout main
# do urgent work
git checkout feature-branch
git stash pop
# hope there are no conflicts...
# rebuild mental context...
# wait for Claude to re-understand everything...
```

This workflow is fragile, slow, and destroys the deep understanding Claude has built about your specific task.

### The Solution: Git Worktrees

Git worktrees allow you to have **multiple working directories** attached to the same repository, each checked out to a different branch. Combined with Claude Code, this enables:

- **Parallel Claude sessions** working on different tasks simultaneously
- **Complete isolation** between different features/bugs
- **Zero context switching** for Claude—each session maintains its deep understanding
- **Instant task switching** for you—just change directories

---

## 2. Understanding Git Worktrees

### What is a Git Worktree?

A Git worktree is a separate working directory linked to the same Git repository. Unlike cloning (which duplicates the entire `.git` directory), worktrees share a single repository database and only create the working files you need.

### Architecture

```
/projects/
├── my-project/              # Main worktree (your original checkout)
│   └── .git/                # The actual Git repository data
│
├── my-project-feature-auth/ # Linked worktree
│   └── .git                 # Text file pointing to main .git
│
├── my-project-bugfix-api/   # Another linked worktree
│   └── .git                 # Text file pointing to main .git
│
└── my-project-refactor/     # Another linked worktree
    └── .git                 # Text file pointing to main .git
```

### Key Properties

| Property | Behavior |
|----------|----------|
| `.git` directory | Shared across all worktrees |
| Working files | Independent per worktree |
| Git history | Shared—commits visible everywhere |
| Remote connections | Shared |
| Branch checkout | Cannot checkout same branch in multiple worktrees |
| File changes | Completely isolated per worktree |

---

## 3. Why Worktrees + Claude Code

### The Context Preservation Problem

When you force Claude Code to context-switch between branches in a single directory, you throw away its most valuable asset: **the deep understanding** it has built about your specific codebase, your patterns, and your goals.

### Benefits of Worktree + Claude Combination

1. **Parallelization**: Run one Claude refactoring your authentication system while another builds a data visualization component—both at full speed.

2. **Context Isolation**: Each Claude session maintains its conversation history and understanding specific to that task.

3. **Increased Impact**: Get different versions of a feature from parallel Claude sessions, then cherry-pick the best one.

4. **Continuous Development**: Keep working in one worktree while Claude runs long tasks in another.

5. **Session Persistence**: Resume any Claude session later—the `/resume` picker shows sessions from all worktrees in the same repository.

### The Developer's Role Shift

With worktrees + Claude Code:
- **Before**: You write code, context-switch constantly
- **After**: You orchestrate multiple AI "team members," review code, and make strategic decisions

You become a **Software Engineering Manager** rather than just a coder.

---

## 4. Basic Setup and Commands

### Creating Worktrees

```bash
# Create a worktree with a NEW branch
git worktree add ../project-feature-auth -b feature/auth

# Create a worktree with an EXISTING branch
git worktree add ../project-bugfix-123 bugfix/issue-123

# Create a worktree from a specific commit/tag
git worktree add ../project-v2-hotfix v2.0.0 -b hotfix/v2-critical

# Create a detached HEAD worktree (for experimentation)
git worktree add -d ../project-experiment
```

### Listing Worktrees

```bash
git worktree list
```

Output:
```
/home/user/projects/my-project         abc1234 [main]
/home/user/projects/my-project-auth    def5678 [feature/auth]
/home/user/projects/my-project-bugfix  ghi9012 [bugfix/issue-123]
```

### Removing Worktrees

```bash
# Remove a worktree (directory must be clean)
git worktree remove ../project-feature-auth

# Force remove (even with uncommitted changes)
git worktree remove --force ../project-feature-auth

# Clean up stale worktree references
git worktree prune
```

### Running Claude in a Worktree

```bash
# Navigate to worktree
cd ../project-feature-auth

# Start Claude Code
claude

# Or start with an initial prompt
claude "Implement OAuth2 authentication for the API"
```

---

## 5. Parallel Development Workflow

### Step-by-Step Workflow

#### Step 1: Create Worktrees for Each Task

```bash
# From your main project directory
git worktree add ../myproject-feature-auth -b feature/user-authentication
git worktree add ../myproject-bugfix-api -b bugfix/api-rate-limiting
git worktree add ../myproject-refactor -b refactor/cleanup-utils
```

#### Step 2: Initialize Each Worktree Environment

```bash
# For each worktree, install dependencies
cd ../myproject-feature-auth
npm install  # or pip install -r requirements.txt, etc.

# Optionally run /init to create CLAUDE.md
claude
/init
```

#### Step 3: Open Multiple Terminal Panes

Using iTerm2, tmux, or VS Code terminals:
- **Pane 1**: `cd ../myproject-feature-auth && claude`
- **Pane 2**: `cd ../myproject-bugfix-api && claude`
- **Pane 3**: `cd ../myproject-refactor && claude`

#### Step 4: Assign Tasks to Each Claude Session

**Pane 1 (Auth Feature)**:
```
Implement OAuth2 authentication with the following requirements:
1. Support Google and GitHub providers
2. Store tokens securely in the database
3. Create middleware for protected routes
4. Write comprehensive tests
```

**Pane 2 (Bug Fix)**:
```
Fix the API rate limiting issue described in GitHub issue #456.
First, use gh issue view 456 to understand the problem,
then implement a solution with proper tests.
```

**Pane 3 (Refactor)**:
```
Refactor the utils/ directory:
1. Remove duplicate code
2. Add TypeScript types
3. Improve error handling
4. Update all imports in the codebase
```

#### Step 5: Cycle Through and Monitor

- Check progress in each pane
- Approve permission requests
- Provide guidance when Claude asks questions
- Review generated code

#### Step 6: Review and Merge

```bash
# In each worktree, create PRs
cd ../myproject-feature-auth
gh pr create --draft --title "feat: OAuth2 authentication"

# Or use Claude to create the PR
# In Claude session:
> Create a PR for these changes with comprehensive description
```

#### Step 7: Cleanup

```bash
# After merging, remove worktrees
git worktree remove ../myproject-feature-auth
git worktree remove ../myproject-bugfix-api
git worktree remove ../myproject-refactor

# Clean up stale references
git worktree prune
```

---

## 6. Automation Scripts and Tools

### Simple Bash Function for Worktree Management

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Quick worktree creation and Claude launch
cw() {
    local branch_name=$1
    local project_name=$(basename $(git rev-parse --show-toplevel))
    local worktree_path="../${project_name}-${branch_name}"
    
    # Create worktree
    git worktree add "$worktree_path" -b "$branch_name" 2>/dev/null || \
    git worktree add "$worktree_path" "$branch_name"
    
    # Navigate and start Claude
    cd "$worktree_path"
    
    # Install dependencies based on project type
    if [ -f "package.json" ]; then
        npm install
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    # Start Claude
    claude
}

# List all worktrees with status
cwl() {
    git worktree list
}

# Remove worktree and optionally delete branch
cwr() {
    local worktree_path=$1
    local branch_name=$(git -C "$worktree_path" branch --show-current)
    
    git worktree remove "$worktree_path"
    
    read -p "Delete branch $branch_name? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git branch -d "$branch_name"
    fi
}
```

Usage:
```bash
# Create worktree and launch Claude
cw feature-auth

# List worktrees
cwl

# Remove worktree
cwr ../myproject-feature-auth
```

### Advanced Worktree Manager Script

Save as `cw` in your PATH:

```bash
#!/bin/bash
# Claude Worktree Manager (cw)
# Usage: cw setup <branch-name> [sparse-path]
#        cw list
#        cw cleanup <branch-name>

set -e

WORKTREE_DIR="gwt"  # Worktrees directory name

cmd=$1
branch=$2
sparse_path=$3

case $cmd in
    setup)
        if [ -z "$branch" ]; then
            echo "Usage: cw setup <branch-name> [sparse-path]"
            exit 1
        fi
        
        worktree_path="$WORKTREE_DIR/$branch"
        
        # Create worktree
        if [ -n "$sparse_path" ]; then
            # Sparse checkout for monorepos
            git worktree add --no-checkout "$worktree_path" -b "$branch"
            cd "$worktree_path"
            git sparse-checkout init --cone
            git sparse-checkout set "$sparse_path"
        else
            git worktree add "$worktree_path" -b "$branch"
            cd "$worktree_path"
        fi
        
        # Copy environment files
        if [ -f "../.env" ]; then
            cp "../.env" ".env"
        fi
        
        # Copy Claude configuration
        if [ -d "../.claude" ]; then
            cp -r "../.claude" ".claude"
        fi
        
        echo "Worktree created at $worktree_path"
        echo "Run: cd $worktree_path && claude"
        ;;
        
    list)
        git worktree list
        ;;
        
    cleanup)
        if [ -z "$branch" ]; then
            echo "Usage: cw cleanup <branch-name>"
            exit 1
        fi
        
        worktree_path="$WORKTREE_DIR/$branch"
        git worktree remove "$worktree_path"
        git branch -d "$branch" 2>/dev/null || true
        echo "Cleaned up $branch"
        ;;
        
    *)
        echo "Claude Worktree Manager"
        echo ""
        echo "Usage:"
        echo "  cw setup <branch-name> [sparse-path]  Create worktree"
        echo "  cw list                                List worktrees"
        echo "  cw cleanup <branch-name>              Remove worktree"
        ;;
esac
```

## 7. Custom Claude Commands for Worktrees

### Create a `/worktree` Command

Create `.claude/commands/worktree.md`:

```markdown
---
argument-hint: branch-name
description: Create a git worktree in a peer directory
---

Create a git worktree in a peer directory.

## Arguments

The argument should be a kebab-case task name (e.g., "auth-feature", "database-migration").

The user passed in: `$ARGUMENTS`

If that text is already kebab case, use it directly as the branch name.
Otherwise come up with a good kebab-case name based on what the user passed in.

## Steps

1. Determine the project name from the current directory
2. Create worktree path as `../<project>-<branch-name>`
3. Run: `git worktree add <path> -b <branch-name>`
4. Copy any `.env` files to the new worktree
5. Copy `.claude` directory if it exists
6. Install dependencies based on project type:
   - If `package.json` exists: `npm install`
   - If `requirements.txt` exists: `pip install -r requirements.txt`
   - If `Cargo.toml` exists: `cargo build`

## Conclusion

Open a new terminal tab in the newly created worktree.
Provide the user with the path and next steps.
```

Usage:
```
/worktree auth-feature
/worktree "implement user login"
```

### Create a `/pr` Command for Worktrees

Create `.claude/commands/pr.md`:

```markdown
---
allowed-tools: Bash(git *), Bash(gh *)
description: Create a pull request from the current worktree
---

Create a pull request for the changes in this worktree.

## Steps

1. Check for uncommitted changes: `git status`
2. If there are changes, commit them with a descriptive message
3. Push the branch: `git push -u origin HEAD`
4. Create a draft PR: `gh pr create --draft`
5. Generate comprehensive PR description including:
   - Summary of changes
   - Testing performed
   - Screenshots if UI changes
   - Breaking changes if any

## After Creation

Provide the PR URL and ask if the user wants to:
- Mark the PR as ready for review
- Request specific reviewers
- Add labels
```

### Create a `/done` Command

Create `.claude/commands/done.md`:

```markdown
---
description: Complete work in this worktree and prepare for merge
---

Finalize work in this worktree.

## Steps

1. Run all tests: detect test framework and execute
2. Run linting/formatting
3. Commit any remaining changes
4. Push to remote
5. Create or update the pull request
6. Provide summary of:
   - Branch name
   - PR URL
   - Files changed
   - Tests passed/failed
   - Next steps for merging
```

---

## 8. Advanced Techniques

### Sparse Checkout for Monorepos

For large monorepos, combine worktrees with sparse checkout to limit Claude's context:

```bash
# Create worktree with NO files initially
git worktree add --no-checkout gwt/api-fix -b fix/api-bug

# Navigate and configure sparse checkout
cd gwt/api-fix
git sparse-checkout init --cone
git sparse-checkout set api/

# Now only the api/ folder exists in this worktree
ls
# api/
```

Benefits:
- **Reduced AI Context**: Claude reads fewer files, responses are more focused
- **Less hallucination**: Smaller context = more precision
- **Faster operations**: `git status`, `git diff` run faster

### Running Multiple Models in Parallel

Use worktrees to compare different approaches:

```bash
# Create worktrees for different experiments
git worktree add ../project-approach-a -b exp/approach-a
git worktree add ../project-approach-b -b exp/approach-b

# In separate terminals, give the same task with different prompts
# Terminal 1:
cd ../project-approach-a && claude
> Implement user search using Elasticsearch

# Terminal 2:
cd ../project-approach-b && claude
> Implement user search using PostgreSQL full-text search

# Compare results and cherry-pick the winner
```

### Batch Processing with Worktrees

For large-scale migrations:

```bash
# Loop to create multiple worktrees with rate limiting
for i in $(seq 1 10); do
    claude --dangerously-skip-permissions --print /worktree
    sleep 300  # Wait 5 minutes between tasks
done
```

### Container Isolation with Worktrees

For maximum isolation, run Claude in containers:

```bash
# Create worktree
git worktree add ../project-experiment -b exp/risky-changes

# Run Claude in a Docker container mounted to that worktree
docker run -it -v $(pwd)/../project-experiment:/workspace \
    anthropic/claude-code:latest \
    claude --dangerously-skip-permissions \
    "Experiment with aggressive refactoring"
```

---

## 9. Merging and Integration

### Integration Branch Strategy

Use an integration branch as a central hub:

```bash
# Create integration branch
git checkout -b integration main

# Merge completed worktree branches
git merge --no-ff feature/auth
git merge --no-ff bugfix/api
git merge --no-ff refactor/utils

# Resolve any conflicts
# Run full test suite
npm test

# If all tests pass, merge to main
git checkout main
git merge integration
```

### Cherry-Picking Specific Commits

When you only want some changes from a worktree:

```bash
# In main worktree, cherry-pick specific commits
git cherry-pick abc1234  # Specific commit from feature branch
git cherry-pick def5678..ghi9012  # Range of commits
```

### Handling Merge Conflicts

Conflicts are inevitable with parallel development:

```bash
# When merging produces conflicts
git merge feature/auth
# Auto-merging src/auth.js
# CONFLICT (content): Merge conflict in src/auth.js

# Options:
# 1. Resolve manually
vim src/auth.js
git add src/auth.js
git commit

# 2. Use Claude to help resolve
claude "Help me resolve the merge conflict in src/auth.js"

# 3. Abort and try different approach
git merge --abort
```

### Post-Merge Cleanup

```bash
# After successful merge, clean up worktrees
git worktree list
git worktree remove ../project-feature-auth
git worktree remove ../project-bugfix-api

# Delete merged branches
git branch -d feature/auth
git branch -d bugfix/api

# Prune stale worktree references
git worktree prune

# Verify cleanup
git worktree list
git branch -a
```

---

## 10. Troubleshooting

### Common Issues and Solutions

#### "fatal: '<branch>' is already checked out"

You cannot checkout the same branch in multiple worktrees.

```bash
# Solution 1: Create a new branch
git worktree add ../project-test -b test-branch

# Solution 2: Use detached HEAD
git worktree add -d ../project-test
```

#### Dependencies Not Found ("Module not found")

Worktrees don't share `node_modules`, `.venv`, or other dependency directories.

```bash
# Solution: Install dependencies in each worktree
cd ../project-feature
npm install  # or pip install, etc.
```

#### Port Already in Use

Running multiple dev servers across worktrees:

```bash
# Start on different ports
PORT=3001 npm run dev
PORT=3002 npm run dev

# Claude usually handles this automatically
```

#### Worktree Directory Deleted Without Cleanup

```bash
# If you manually deleted a worktree directory
git worktree prune

# Verify
git worktree list
```

#### Submodules in Worktrees

Worktrees don't automatically initialize submodules:

```bash
cd ../project-worktree
git submodule update --init --recursive
```

#### `.env` Files Missing

Environment files aren't copied automatically:

```bash
# Copy manually
cp ../main-project/.env ./.env

# Or add to your worktree script
```

### Diagnostic Commands

```bash
# Check worktree health
git worktree list --porcelain

# Verify worktree linkage
cat .git  # Should show: gitdir: /path/to/main/.git/worktrees/<name>

# Check for stale entries
git worktree prune --dry-run

# See worktree configuration
git config --list | grep worktree
```

---

## 11. Best Practices

### Do's ✅

1. **Use descriptive branch/directory names**
   ```bash
   # Good
   git worktree add ../myproject-feature-user-auth -b feature/user-auth
   
   # Bad
   git worktree add ../test -b test
   ```

2. **Run `/init` in each worktree session**
   ```
   # In Claude session
   /init
   ```

3. **Install dependencies immediately after creating worktree**
   ```bash
   cd ../new-worktree && npm install && claude
   ```

4. **Create draft PRs for Claude-generated code**
   ```bash
   gh pr create --draft
   ```

5. **Limit to 2-3 active worktrees**
   - Your review bandwidth is the bottleneck, not Claude's speed
   - Token consumption scales non-linearly

6. **Name Claude sessions descriptively**
   - Use `/resume` to manage multiple sessions
   - Press `R` in the picker to rename

7. **Copy `.env` and `.claude` configurations**
   ```bash
   cp ../.env . && cp -r ../.claude .
   ```

8. **Clean up worktrees after merging**
   ```bash
   git worktree remove ../completed-worktree
   git worktree prune
   ```

### Don'ts ❌

1. **Don't run 5+ parallel agents**
   - Coordination overhead exceeds benefits
   - API rate limits become an issue

2. **Don't forget dependency installation**
   - Worktrees are like fresh clones for dependencies

3. **Don't use `--force` for worktree removal unless necessary**
   - You might lose uncommitted work

4. **Don't expect worktrees to share runtime state**
   - Dev servers, build caches, etc. are independent

5. **Don't mix worktree concerns**
   - Keep each worktree focused on one task

6. **Don't leave stale worktrees around**
   - They consume disk space and cause confusion

### Directory Organization

Recommended structure:

```
~/projects/
├── myproject/                    # Main checkout
│   ├── .git/
│   ├── .claude/
│   └── src/
│
├── worktrees/                    # Or adjacent to main
│   ├── myproject-feature-auth/
│   ├── myproject-bugfix-api/
│   └── myproject-refactor/
│
└── .worktrees/                   # Alternative: hidden in project
```

Or use a `.worktrees/` directory inside the project:

```bash
mkdir .worktrees
echo ".worktrees/" >> .gitignore
git worktree add .worktrees/feature-auth -b feature/auth
```

---

## 12. Real-World Examples

### Example 1: Feature Development + Urgent Bug Fix

**Scenario**: You're building a new dashboard feature when a critical production bug is reported.

```bash
# Initial setup: working on dashboard
cd ~/projects/myapp
claude
> Building the analytics dashboard...

# URGENT: Production bug reported!

# Create worktree for bug fix (don't interrupt dashboard work)
git worktree add ../myapp-hotfix -b hotfix/payment-bug

# New terminal
cd ../myapp-hotfix
npm install
claude
> Fix the payment double-charge bug. Use gh issue view 789 for details.

# Claude fixes the bug while your dashboard session is untouched
# After fix is done:
gh pr create --title "fix: prevent payment double-charge"

# Back to dashboard (everything still there!)
cd ../myapp
# Continue working...

# Clean up after hotfix is merged
git worktree remove ../myapp-hotfix
```

### Example 2: Parallel Feature Comparison

**Scenario**: Implement the same feature two different ways and compare.

```bash
# Create two worktrees
git worktree add ../myapp-search-elastic -b exp/search-elasticsearch
git worktree add ../myapp-search-postgres -b exp/search-postgres

# Terminal 1
cd ../myapp-search-elastic && npm install && claude
> Implement product search using Elasticsearch.
> Focus on fuzzy matching and performance.

# Terminal 2
cd ../myapp-search-postgres && npm install && claude
> Implement product search using PostgreSQL full-text search.
> Focus on simplicity and no external dependencies.

# After both complete, compare:
# - Performance benchmarks
# - Code complexity
# - Maintenance burden

# Merge the winner
git checkout main
git merge exp/search-postgres  # Example: chose PostgreSQL

# Clean up
git worktree remove ../myapp-search-elastic
git worktree remove ../myapp-search-postgres
git branch -d exp/search-elasticsearch
git branch -d exp/search-postgres
```

### Example 3: Code Review in Isolation

**Scenario**: Review a colleague's PR without affecting your current work.

```bash
# Fetch the PR branch
git fetch origin

# Create worktree for review
git worktree add .worktrees/review-pr-423 origin/feature/new-dashboard

# Open in IDE and run
cd .worktrees/review-pr-423
npm install
npm test
npm run dev

# Use Claude to analyze
claude
> Review this code for:
> 1. Security vulnerabilities
> 2. Performance issues
> 3. Code style violations
> 4. Missing test coverage

# After review, clean up
cd ../..
git worktree remove .worktrees/review-pr-423
```

### Example 4: Large-Scale Migration

**Scenario**: Migrate 100 React components to a new design system.

```bash
# Create task list
claude -p "List all React components that need migration" > components.txt

# Process in batches with worktrees
for batch in {1..5}; do
    git worktree add "../myapp-migration-$batch" -b "migrate/batch-$batch"
    
    # Start Claude in background
    cd "../myapp-migration-$batch"
    npm install
    claude --dangerously-skip-permissions \
        "Migrate components $((batch*20-19)) through $((batch*20)) from components.txt to the new design system"
    
    cd ..
done

# Monitor progress in each worktree
# Merge completed batches
# Clean up
```

### Example 5: Monorepo Sparse Checkout

**Scenario**: Fix a backend API bug without loading the entire frontend.

```bash
# Create sparse worktree for backend only
git worktree add --no-checkout gwt/api-fix -b fix/api-bug

cd gwt/api-fix
git sparse-checkout init --cone
git sparse-checkout set packages/api/ packages/shared/

# Now only backend packages are checked out
npm install
claude
> Fix the rate limiting bug in packages/api/src/middleware/rateLimit.ts

# Claude has focused context, faster responses, better accuracy
```

---

## Quick Reference Card

### Essential Commands

```bash
# Create worktree with new branch
git worktree add <path> -b <branch>

# Create worktree with existing branch
git worktree add <path> <branch>

# List worktrees
git worktree list

# Remove worktree
git worktree remove <path>

# Clean up stale references
git worktree prune

# Navigate and start Claude
cd <worktree-path> && claude
```

### Workflow Checklist

```
□ Create worktree with descriptive name
□ Install dependencies
□ Copy .env and .claude configs
□ Run /init in Claude session
□ Assign focused task to Claude
□ Create draft PR when done
□ Review and merge
□ Remove worktree
□ Delete branch if merged
□ Run git worktree prune
```

### Session Management

```bash
# Resume sessions (shows all worktrees)
/resume

# Link session to PR
claude --from-pr 123

# List Claude sessions
# (In /resume picker, sessions show worktree metadata)
```

---

## Resources

- [Official Git Worktree Documentation](https://git-scm.com/docs/git-worktree)
- [Claude Code Common Workflows](https://code.claude.com/docs/en/common-workflows)
- [Anthropic's Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [worktree-workflow Repository](https://github.com/anthropics/worktree-workflow)